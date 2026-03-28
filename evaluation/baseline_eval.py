import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
from datasets import load_dataset
from tqdm import tqdm

from evaluation.metrics import (
    compute_wer,
    compute_cer,
    analyze_failures,
    compute_stratified_wer
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA debug check
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if not torch.cuda.is_available():
    print("WARNING: Running on CPU — install the CUDA build of PyTorch for GPU acceleration.")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

HF_TOKEN = os.getenv("HF_TOKEN")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

MODELS = {
    "whisper_small": {
        "type": "whisper",
        "name": "openai/whisper-small",
        "language": "ta",
        "task": "transcribe"
    },
    "whisper_tamil": {
        "type": "whisper",
        "name": "vasista22/whisper-tamil-medium",
        "language": "ta",
        "task": "transcribe"
    },
    "wav2vec2_tamil": {
        "type": "wav2vec2",
        "name": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
        "language": "ta",
        "task": "transcribe"
    }
}


def load_whisper_model(model_name: str):
    """Load Whisper model and processor."""
    logger.info(f"Loading Whisper model: {model_name}")
    processor = WhisperProcessor.from_pretrained(
        model_name, token=HF_TOKEN
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    # Normalize generation config for compatibility with transformers ≥ 4.36.
    # New-style models (e.g. openai/whisper-small) have lang_to_id in their
    # generation_config and support the language= kwarg to generate().
    # Old-style fine-tuned models (e.g. vasista22) lack lang_to_id and use
    # forced_decoder_ids instead — Tamil is already baked in there, so we
    # leave that config alone and call generate() without language/task kwargs.
    if hasattr(model.generation_config, "lang_to_id"):
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        model.generation_config.language = "tamil"
        model.generation_config.task = "transcribe"
    return processor, model


def load_wav2vec2_model(model_name: str):
    """Load Wav2Vec2 model and processor."""
    logger.info(f"Loading Wav2Vec2 model: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(
        model_name, token=HF_TOKEN
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    return processor, model


def transcribe_whisper(
    audio: np.ndarray,
    processor,
    model,
    language: str = "ta"
) -> str:
    """Transcribe audio using Whisper model."""
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    if DEVICE == "cuda":
        inputs = inputs.half()

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs,
            max_new_tokens=256
        )

    return processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()


def transcribe_wav2vec2(
    audio: np.ndarray,
    processor,
    model
) -> str:
    """Transcribe audio using Wav2Vec2 model."""
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values.to(DEVICE)

    if DEVICE == "cuda":
        inputs = inputs.half()

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].strip()


def evaluate_model(
    model_key: str,
    model_config: dict,
    test_samples: list,
    max_samples: Optional[int] = None
) -> dict:
    """
    Run a single model on all test samples.
    Returns structured results dict matching baseline_wer.json schema.
    """
    model_type = model_config["type"]
    model_name = model_config["name"]

    if model_type == "whisper":
        processor, model = load_whisper_model(model_name)
        transcribe_fn = lambda audio: transcribe_whisper(
            audio, processor, model, model_config["language"]
        )
    elif model_type == "wav2vec2":
        processor, model = load_wav2vec2_model(model_name)
        transcribe_fn = lambda audio: transcribe_wav2vec2(
            audio, processor, model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    samples = test_samples
    if max_samples:
        samples = test_samples[:max_samples]

    per_sample_results = []
    errors = 0

    for sample in tqdm(samples, desc=f"Evaluating {model_key}"):
        try:
            audio = np.array(sample["audio"], dtype=np.float32)
            reference = sample["transcript"]
            hypothesis = transcribe_fn(audio)
            analysis = analyze_failures(reference, hypothesis)
            analysis["reference"] = reference
            analysis["hypothesis"] = hypothesis
            per_sample_results.append(analysis)
        except Exception as e:
            logger.warning(f"Sample failed: {e}")
            errors += 1
            continue

    stratified = compute_stratified_wer(per_sample_results)

    result = {
        "model_name": model_name,
        "model_key": model_key,
        "device": DEVICE,
        "total_samples": len(per_sample_results),
        "errors": errors,
        "overall_wer": stratified["overall_wer"],
        "monolingual_tamil_wer": stratified["monolingual_tamil_wer"],
        "monolingual_english_wer": stratified["monolingual_english_wer"],
        "code_switched_wer": stratified["code_switched_wer"],
        "failure_breakdown": stratified["failure_breakdown"]
    }

    logger.info(f"\n=== {model_key} Results ===")
    logger.info(f"Overall WER:        {result['overall_wer']}")
    logger.info(f"Monolingual Tamil:  {result['monolingual_tamil_wer']}")
    logger.info(f"Monolingual English:{result['monolingual_english_wer']}")
    logger.info(f"Code-switched WER:  {result['code_switched_wer']}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return result


def run_all_baselines(
    test_samples: list,
    max_samples: Optional[int] = 100,
    models_to_run: Optional[list] = None
):
    """
    Run all baseline models and save results.
    Set max_samples=None to run on full test set.
    """
    if models_to_run is None:
        models_to_run = list(MODELS.keys())

    all_results = {}

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.warning(f"Unknown model: {model_key}, skipping.")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {model_key}")
        logger.info(f"{'='*50}")

        try:
            result = evaluate_model(
                model_key,
                MODELS[model_key],
                test_samples,
                max_samples=max_samples
            )
        except Exception as e:
            logger.error(f"Model {model_key} failed — skipping. Reason: {e}")
            continue

        all_results[model_key] = result

        save_path = RESULTS_DIR / f"{model_key}_wer.json"
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved results to {save_path}")

    combined_path = RESULTS_DIR / "baseline_wer_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {combined_path}")

    print_comparison_table(all_results)
    return all_results


def print_comparison_table(results: dict):
    """Print a clean WER comparison table to console."""
    print("\n" + "="*80)
    print("BASELINE WER COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Overall':>10} {'Mono-Tamil':>12} "
          f"{'Mono-English':>14} {'Code-Switch':>13}")
    print("-"*80)
    for model_key, r in results.items():
        print(
            f"{model_key:<20} "
            f"{str(r['overall_wer']):>10} "
            f"{str(r['monolingual_tamil_wer']):>12} "
            f"{str(r['monolingual_english_wer']):>14} "
            f"{str(r['code_switched_wer']):>13}"
        )
    print("="*80 + "\n")


if __name__ == "__main__":
    from data.prepare_dataset import (
        authenticate_hf,
        load_indicvoices_tamil,
        build_dataset_splits
    )

    authenticate_hf()
    samples = load_indicvoices_tamil(max_samples=300)
    splits = build_dataset_splits(samples)
    test_samples = splits["test"]

    logger.info(f"Test set size: {len(test_samples)} samples")

    run_all_baselines(
        test_samples,
        max_samples=50,
        models_to_run=["whisper_medium"]
    )

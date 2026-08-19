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
from tqdm import tqdm

from data.corpus import (
    CORPUS_DIR,
    SPLIT_NAMES,
    atomic_write_json,
    compute_corpus_id,
    load_split,
    split_corpus_id,
)
from evaluation.metrics import (
    analyze_failures,
    compute_stratified_wer,
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
    if max_samples is not None:
        samples = test_samples[:max_samples]

    per_sample_results = []
    scored_uids = []
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
            if sample.get("uid"):
                scored_uids.append(sample["uid"])
        except Exception as e:
            logger.warning(f"Sample failed: {e}")
            errors += 1
            continue

    stratified = compute_stratified_wer(per_sample_results)

    # Identify the exact data this score describes. `corpus_id` covers the
    # samples that actually contributed (a cap, or a sample that failed to
    # transcribe, changes it); `split_corpus_id` is the full frozen split they
    # were drawn from. Without these two fields a results file cannot honestly
    # be compared to any other.
    corpus_id = compute_corpus_id(scored_uids) if scored_uids else None
    split_names = {s.get("split") for s in samples if s.get("split")}
    split_ids = {s.get("corpus_id") for s in samples if s.get("corpus_id")}

    result = {
        "model_name": model_name,
        "model_key": model_key,
        "device": DEVICE,
        "split": split_names.pop() if len(split_names) == 1 else None,
        "corpus_id": corpus_id,
        "split_corpus_id": split_ids.pop() if len(split_ids) == 1 else None,
        "total_samples": len(per_sample_results),
        "errors": errors,
        "overall_wer": stratified["overall_wer"],
        "monolingual_tamil_wer": stratified["monolingual_tamil_wer"],
        "monolingual_english_wer": stratified["monolingual_english_wer"],
        "code_switched_wer": stratified["code_switched_wer"],
        "failure_breakdown": stratified["failure_breakdown"]
    }

    logger.info(f"\n=== {model_key} Results ===")
    logger.info(f"Corpus:             {result['corpus_id']} "
                f"(split {result['split']}, {result['total_samples']} samples)")
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
    models_to_run: Optional[list] = None,
    results_dir: Optional[Path] = None,
):
    """
    Run all baseline models and save results.
    Set max_samples=None to run on full test set.

    Raises ValueError on an unknown model key rather than skipping it — a typo
    here previously produced an empty result set that then overwrote the
    committed baseline_wer_all.json.
    """
    if models_to_run is None:
        models_to_run = list(MODELS.keys())
    # Resolved at call time, not import time, so tests and capped smoke runs can
    # redirect output without overwriting the committed results.
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    unknown = [k for k in models_to_run if k not in MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {unknown}. "
            f"Known models: {sorted(MODELS)}"
        )

    all_results = {}

    for model_key in models_to_run:
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

        save_path = results_dir / f"{model_key}_wer.json"
        atomic_write_json(save_path, result)
        logger.info(f"Saved results to {save_path}")

    # Never overwrite the combined file with an empty or partial result set —
    # a failed run must leave the previously committed results intact.
    combined_path = results_dir / "baseline_wer_all.json"
    if not all_results:
        logger.error(
            "No model produced results; leaving %s untouched.", combined_path
        )
        return all_results

    if len(all_results) < len(models_to_run):
        failed = [k for k in models_to_run if k not in all_results]
        logger.error(
            "Only %d/%d models succeeded (failed: %s); leaving %s untouched. "
            "Re-run once the failures are fixed, or pass models_to_run "
            "explicitly to write a partial comparison on purpose.",
            len(all_results), len(models_to_run), failed, combined_path,
        )
        print_comparison_table(all_results)
        return all_results

    # Merge into whatever is already there rather than replacing it. Running a
    # single model used to rewrite the combined file down to one row, silently
    # discarding the other models' results.
    merged = {}
    if combined_path.exists():
        try:
            with open(combined_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(
                "Could not read %s (%s); leaving it untouched.", combined_path, e
            )
            print_comparison_table(all_results)
            return all_results
    malformed = [k for k, v in merged.items() if not isinstance(v, dict)]
    if malformed:
        logger.error(
            "%s is malformed (non-object rows: %s); leaving it untouched.",
            combined_path, malformed,
        )
        print_comparison_table(all_results)
        return all_results

    merged.update(all_results)

    # The combined file is read as a comparison table. Rows scored on different
    # data are exactly the defect this pipeline was retracted for, so refuse to
    # write rather than produce a misleading artifact. This covers rows carried
    # over from an earlier run, which is how the original comparison was formed.
    ids = {k: r.get("corpus_id") for k, r in merged.items()}
    if len(set(ids.values())) > 1:
        logger.error(
            "Refusing to write %s: rows were scored on different data. "
            "corpus_id per model: %s. Re-run every model on one frozen split.",
            combined_path, ids,
        )
        print_comparison_table(all_results)
        return all_results

    atomic_write_json(combined_path, merged)
    logger.info("Wrote %d model(s) to %s", len(merged), combined_path)

    print_comparison_table(all_results)
    return all_results


def print_comparison_table(results: dict):
    """Print a clean WER comparison table to console."""
    print("\n" + "="*80)
    print("BASELINE WER COMPARISON")
    ids = {r.get("corpus_id") for r in results.values()}
    if len(ids) == 1:
        cid = ids.pop()
        print(f"corpus_id: {cid}  (all rows scored on the same samples)")
    elif len(ids) > 1:
        print("WARNING: rows were scored on DIFFERENT data and are not comparable.")
        for key, r in results.items():
            print(f"  {key:<20} corpus_id {r.get('corpus_id')}")
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


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m evaluation.baseline_eval",
        description="Evaluate baseline ASR models on the Tanglish test split.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="KEY",
        help=f"Model keys to run (default: all). Choices: {sorted(MODELS)}",
    )
    parser.add_argument(
        "--split", default="test", choices=list(SPLIT_NAMES),
        help="Frozen corpus split to evaluate (default: test).",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=CORPUS_DIR,
        help=f"Frozen corpus root (default: {CORPUS_DIR}).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap evaluated test samples (default: no cap). A capped run scores "
             "a subset, so it gets its own corpus_id and is not comparable to a "
             "full run. Pair it with --results-dir.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help=f"Where to write results (default: {RESULTS_DIR}). Point smoke runs "
             f"elsewhere so they do not overwrite committed results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the models that would run, then exit without writing.",
    )
    args = parser.parse_args(argv)

    models_to_run = args.models or list(MODELS.keys())
    unknown = [k for k in models_to_run if k not in MODELS]
    if unknown:
        parser.error(
            f"unknown model key(s): {unknown}. Known models: {sorted(MODELS)}"
        )

    if args.dry_run:
        print(f"Would evaluate {len(models_to_run)} model(s):")
        for key in models_to_run:
            print(f"  {key:<16} {MODELS[key]['name']}")
        try:
            cid = split_corpus_id(args.split, corpus_dir=args.corpus_dir)
            print(f"On split {args.split!r} of {args.corpus_dir} (corpus_id {cid})")
        except FileNotFoundError as e:
            print(f"No frozen corpus: {e}")
        print("Dry run — no files written.")
        return 0

    # Read the frozen split rather than rebuilding from HuggingFace. Rebuilding
    # here is what produced two different test sets for two sets of models.
    try:
        test_samples = load_split(args.split, corpus_dir=args.corpus_dir)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info(
        "Loaded %d samples from frozen split %r (corpus_id %s)",
        len(test_samples), args.split,
        test_samples[0]["corpus_id"] if test_samples else "empty",
    )

    results = run_all_baselines(
        test_samples,
        max_samples=args.max_samples,
        models_to_run=models_to_run,
        results_dir=args.results_dir,
    )
    if len({r.get("corpus_id") for r in results.values()}) > 1:
        return 1
    return 0 if len(results) == len(models_to_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())

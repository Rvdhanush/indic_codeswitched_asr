import os
import torch
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from langdetect import detect_langs, DetectorFactory
from huggingface_hub import login
from pathlib import Path
from typing import Optional
import logging

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
DATA_DIR = Path("data/processed")


def authenticate_hf():
    """Login to HuggingFace using token from .env"""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Add it to your .env file.")
    login(token=HF_TOKEN)
    logger.info("HuggingFace authentication successful.")


def resample_audio(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16kHz mono."""
    if orig_sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio.astype(np.float32)


def chunk_audio(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE
) -> list:
    """Split audio into max 30-second chunks."""
    max_samples = MAX_AUDIO_SECONDS * sr
    if len(audio) <= max_samples:
        return [audio]
    chunks = []
    for start in range(0, len(audio), max_samples):
        chunk = audio[start:start + max_samples]
        if len(chunk) > sr:  # skip chunks shorter than 1 second
            chunks.append(chunk)
    return chunks


def detect_language_mix(text: str) -> dict:
    """Detect language composition of transcript text."""
    try:
        langs = detect_langs(text)
        return {str(l.lang): round(l.prob, 3) for l in langs}
    except Exception:
        return {"unknown": 1.0}


def tag_segment_type(text: str) -> str:
    """Tag segment as monolingual_tamil, monolingual_english, or code_switched."""
    lang_mix = detect_language_mix(text)
    en_prob = lang_mix.get("en", 0.0)
    ta_prob = lang_mix.get("ta", 0.0)
    if en_prob > 0.85:
        return "monolingual_english"
    elif ta_prob > 0.85:
        return "monolingual_tamil"
    else:
        return "code_switched"


def count_switch_points(text: str) -> int:
    """Count number of language switch points in transcript."""
    words = text.split()
    if len(words) < 2:
        return 0
    switches = 0
    prev_lang = None
    for word in words:
        try:
            langs = detect_langs(word)
            curr_lang = langs[0].lang if langs else "unknown"
        except Exception:
            curr_lang = "unknown"
        if prev_lang and curr_lang != prev_lang and curr_lang != "unknown":
            switches += 1
        prev_lang = curr_lang
    return switches


def preprocess_sample(sample: dict) -> Optional[dict]:
    """
    Process a single dataset sample.
    Returns None if sample should be skipped.
    """
    try:
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        transcript = sample.get("text", sample.get("transcript", ""))

        if not transcript or len(transcript.strip()) < 3:
            return None

        audio = resample_audio(audio_array, sr)
        duration = len(audio) / TARGET_SAMPLE_RATE

        if duration > MAX_AUDIO_SECONDS:
            audio = audio[:MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE]
            duration = MAX_AUDIO_SECONDS

        segment_type = tag_segment_type(transcript)
        switch_count = count_switch_points(transcript)
        lang_mix = detect_language_mix(transcript)

        return {
            "audio": audio,
            "transcript": transcript.strip(),
            "segment_type": segment_type,
            "switch_count": switch_count,
            "lang_mix_en": lang_mix.get("en", 0.0),
            "lang_mix_ta": lang_mix.get("ta", 0.0),
            "duration_seconds": round(duration, 2),
            "sample_rate": TARGET_SAMPLE_RATE,
        }
    except Exception as e:
        logger.warning(f"Skipping sample due to error: {e}")
        return None


def load_indicvoices_tamil(
    max_samples: int = 1500,
    streaming: bool = True
) -> list:
    """
    Load Tamil samples from IndicVoices dataset.
    Uses streaming to avoid downloading full dataset.
    """
    logger.info("Loading IndicVoices Tamil dataset...")
    try:
        dataset = load_dataset(
            "ai4bharat/indicvoices",
            "ta",
            split="train",
            streaming=streaming,
            token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to load IndicVoices: {e}")
        logger.info("Try accepting the dataset license at: "
                    "https://huggingface.co/datasets/ai4bharat/indicvoices")
        raise

    samples = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        processed = preprocess_sample(sample)
        if processed:
            samples.append(processed)
        if len(samples) % 100 == 0 and len(samples) > 0:
            logger.info(f"Processed {len(samples)} samples...")

    logger.info(f"Loaded {len(samples)} Tamil samples from IndicVoices.")
    return samples


def load_lahaja(max_samples: int = 500) -> list:
    """Load Hindi samples from Lahaja dataset."""
    logger.info("Loading Lahaja dataset...")
    try:
        dataset = load_dataset(
            "ai4bharat/Lahaja",
            split="test",
            token=HF_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to load Lahaja: {e}")
        raise

    samples = []
    for sample in dataset:
        if len(samples) >= max_samples:
            break
        processed = preprocess_sample(sample)
        if processed:
            samples.append(processed)

    logger.info(f"Loaded {len(samples)} Hindi samples from Lahaja.")
    return samples


def build_dataset_splits(
    samples: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> DatasetDict:
    """
    Split samples into train/val/test sets.
    Stratified by segment_type.
    """
    from sklearn.model_selection import train_test_split

    labels = [s["segment_type"] for s in samples]

    train_samples, temp_samples = train_test_split(
        samples, test_size=(1 - train_ratio),
        stratify=labels, random_state=42
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    temp_labels = [s["segment_type"] for s in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples, test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels, random_state=42
    )

    def to_hf_dataset(sample_list):
        return Dataset.from_dict({
            key: [s[key] for s in sample_list]
            for key in sample_list[0].keys()
            if key != "audio"
        })

    logger.info(
        f"Split sizes — Train: {len(train_samples)}, "
        f"Val: {len(val_samples)}, Test: {len(test_samples)}"
    )

    return {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }


def print_dataset_stats(samples: list):
    """Print distribution of segment types in dataset."""
    from collections import Counter
    types = Counter(s["segment_type"] for s in samples)
    total = len(samples)
    print("\n=== Dataset Statistics ===")
    for seg_type, count in types.items():
        print(f"  {seg_type}: {count} ({100*count/total:.1f}%)")
    print(f"  Total: {total} samples")

    code_switched = [s for s in samples if s["segment_type"] == "code_switched"]
    if code_switched:
        avg_switches = np.mean([s["switch_count"] for s in code_switched])
        print(f"  Avg switch points (code-switched): {avg_switches:.2f}")
    print("==========================\n")


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    authenticate_hf()

    tamil_samples = load_indicvoices_tamil(max_samples=1500)
    print_dataset_stats(tamil_samples)

    splits = build_dataset_splits(tamil_samples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import json
    for split_name, split_samples in splits.items():
        save_path = DATA_DIR / f"{split_name}_metadata.json"
        metadata = [
            {k: v for k, v in s.items() if k != "audio"}
            for s in split_samples
        ]
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {split_name} metadata to {save_path}")

    print("Dataset preparation complete.")

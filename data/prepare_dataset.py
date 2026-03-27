"""
Dataset preparation for Tamil-English code-switched ASR.

Strategy: synthetic code-switching
  - Monolingual Tamil  : SPRINGLab/IndicVoices-R_Tamil
  - Monolingual English: librispeech_asr (clean, train.100)
  - Code-switched      : Tamil segment + 0.1s silence + English segment
                         (concatenated audio, mixed transcript)

Target distribution (200 samples default):
  80  code_switched
  70  monolingual_tamil
  50  monolingual_english

Public API (unchanged — baseline_eval.py and train.py depend on these):
  authenticate_hf()
  load_indicvoices_tamil(max_samples, streaming) -> list
  build_dataset_splits(samples, train_ratio, val_ratio) -> dict
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import librosa
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from langdetect import detect_langs, DetectorFactory
from sklearn.model_selection import train_test_split

load_dotenv()

# Raise HuggingFace download timeout before any dataset is loaded.
# Default is 10s — too short for large parquet shards on slow connections.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN        = os.getenv("HF_TOKEN")
TARGET_SR       = 16_000
MIN_DURATION_S  = 2.0
MAX_DURATION_S  = 8.0       # per segment before concatenation
MAX_AUDIO_S     = 30.0      # hard cap on any single sample
SILENCE_S       = 0.1
DATA_DIR        = Path("data/processed")

# Target counts
N_CODE_SWITCHED      = 80
N_MONO_TAMIL         = 70
N_MONO_ENGLISH       = 50

# Load overhead multiplier — accounts for samples filtered by duration.
# Keep low: both datasets have mostly valid durations, so 1.3× is enough.
_OVERSAMPLE = 1.3


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def authenticate_hf():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set. Add it to your .env file.")
    login(token=HF_TOKEN)
    logger.info("HuggingFace authentication successful.")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to TARGET_SR, collapse to mono, return float32."""
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def _duration(audio: np.ndarray) -> float:
    return len(audio) / TARGET_SR


def _trim_to_window(audio: np.ndarray, max_s: float = MAX_DURATION_S) -> np.ndarray:
    """Trim audio to max_s seconds from the start."""
    max_samples = int(max_s * TARGET_SR)
    return audio[:max_samples]


def _silence(s: float = SILENCE_S) -> np.ndarray:
    return np.zeros(int(s * TARGET_SR), dtype=np.float32)


# ---------------------------------------------------------------------------
# Language helpers (unchanged from original — used by metrics.py too)
# ---------------------------------------------------------------------------

def detect_language_mix(text: str) -> dict:
    try:
        langs = detect_langs(text)
        return {str(l.lang): round(l.prob, 3) for l in langs}
    except Exception:
        return {"unknown": 1.0}


def tag_segment_type(text: str) -> str:
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
    words = text.split()
    if len(words) < 2:
        return 0
    switches, prev_lang = 0, None
    for word in words:
        try:
            curr_lang = detect_langs(word)[0].lang
        except Exception:
            curr_lang = "unknown"
        if prev_lang and curr_lang != prev_lang and curr_lang != "unknown":
            switches += 1
        prev_lang = curr_lang
    return switches


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _load_tamil_segments(n: int) -> list:
    """
    Load n valid Tamil segments from SPRINGLab/IndicVoices-R_Tamil.
    Each segment is 2–8 seconds after resampling to 16kHz.
    Returns list of {"audio": np.ndarray, "transcript": str}.
    """
    logger.info(f"Loading {n} Tamil segments from IndicVoices-R_Tamil...")
    ds = load_dataset(
        "SPRINGLab/IndicVoices-R_Tamil",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )
    segments, skipped = [], 0
    for sample in ds:
        if len(segments) >= n:
            break
        try:
            raw = np.array(sample["audio"]["array"], dtype=np.float32)
            sr  = sample["audio"]["sampling_rate"]
            audio = _resample(raw, sr)
            audio = _trim_to_window(audio, MAX_DURATION_S)
            dur = _duration(audio)
            if dur < MIN_DURATION_S:
                skipped += 1
                continue
            transcript = (
                sample.get("verbatim") or
                sample.get("normalized") or
                sample.get("text") or ""
            ).strip()
            if len(transcript) < 3:
                skipped += 1
                continue
            segments.append({"audio": audio, "transcript": transcript})
        except Exception as e:
            logger.debug(f"Tamil sample skipped: {e}")
            skipped += 1
    logger.info(f"Tamil: collected {len(segments)} segments, skipped {skipped}")
    return segments


def _load_english_segments(n: int) -> list:
    """
    Load n valid English segments from librispeech_asr (clean/train.100).
    Each segment is 2–8 seconds.
    Returns list of {"audio": np.ndarray, "transcript": str}.
    """
    logger.info(f"Loading {n} English segments from librispeech_asr...")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        streaming=True,
        trust_remote_code=True,
    )
    segments, skipped = [], 0
    for sample in ds:
        if len(segments) >= n:
            break
        try:
            raw = np.array(sample["audio"]["array"], dtype=np.float32)
            sr  = sample["audio"]["sampling_rate"]
            audio = _resample(raw, sr)
            audio = _trim_to_window(audio, MAX_DURATION_S)
            dur = _duration(audio)
            if dur < MIN_DURATION_S:
                skipped += 1
                continue
            transcript = sample.get("text", "").strip().lower()
            if len(transcript) < 3:
                skipped += 1
                continue
            segments.append({"audio": audio, "transcript": transcript})
        except Exception as e:
            logger.debug(f"English sample skipped: {e}")
            skipped += 1
    logger.info(f"English: collected {len(segments)} segments, skipped {skipped}")
    return segments


# ---------------------------------------------------------------------------
# Synthetic code-switching
# ---------------------------------------------------------------------------

def _make_cs_sample(tamil_seg: dict, english_seg: dict) -> dict:
    """
    Concatenate Tamil + silence + English into one code-switched sample.
    Combined transcript uses Tamil text + English text separated by a space.
    """
    audio = np.concatenate([
        tamil_seg["audio"],
        _silence(SILENCE_S),
        english_seg["audio"],
    ])
    # Hard cap at MAX_AUDIO_S
    max_samples = int(MAX_AUDIO_S * TARGET_SR)
    audio = audio[:max_samples]

    transcript = f"{tamil_seg['transcript']} {english_seg['transcript']}"
    return _build_sample(audio, transcript, segment_type="code_switched", switch_count=1)


def _build_sample(
    audio: np.ndarray,
    transcript: str,
    segment_type: str,
    switch_count: Optional[int] = None,
) -> dict:
    """Package a processed audio array + transcript into the standard sample dict."""
    if switch_count is None:
        switch_count = count_switch_points(transcript)
    lang_mix = detect_language_mix(transcript)
    return {
        "audio":            audio,
        "transcript":       transcript,
        "segment_type":     segment_type,
        "switch_count":     switch_count,
        "lang_mix_en":      lang_mix.get("en", 0.0),
        "lang_mix_ta":      lang_mix.get("ta", 0.0),
        "duration_seconds": round(_duration(audio), 2),
        "sample_rate":      TARGET_SR,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_indicvoices_tamil(
    max_samples: int = 200,
    streaming: bool = True,          # kept for API compatibility
) -> list:
    """
    Build a mixed Tamil-English dataset with synthetic code-switching.

    Distribution (scaled from max_samples=200 defaults):
      40 % code_switched   (Tamil+English concatenated)
      35 % monolingual_tamil
      25 % monolingual_english

    Args:
        max_samples: total samples to return
        streaming:   unused (kept for API compat with baseline_eval / train)

    Returns:
        List of sample dicts with keys:
          audio, transcript, segment_type, switch_count,
          lang_mix_en, lang_mix_ta, duration_seconds, sample_rate
    """
    scale = max_samples / 200
    n_cs  = max(1, int(N_CODE_SWITCHED  * scale))
    n_ta  = max(1, int(N_MONO_TAMIL     * scale))
    n_en  = max(1, int(N_MONO_ENGLISH   * scale))

    # Load source segments (with overhead to cover filtered samples)
    n_ta_needed = int((n_ta + n_cs) * _OVERSAMPLE)
    n_en_needed = int((n_en + n_cs) * _OVERSAMPLE)

    tamil_pool   = _load_tamil_segments(n_ta_needed)
    english_pool = _load_english_segments(n_en_needed)

    if len(tamil_pool) < n_ta + n_cs:
        logger.warning(
            f"Only {len(tamil_pool)} Tamil segments available, "
            f"needed {n_ta + n_cs}. Reducing targets."
        )
        n_ta = max(0, len(tamil_pool) - n_cs)
        n_cs = min(n_cs, len(tamil_pool) - n_ta)

    if len(english_pool) < n_en + n_cs:
        logger.warning(
            f"Only {len(english_pool)} English segments available, "
            f"needed {n_en + n_cs}. Reducing targets."
        )
        n_en = max(0, len(english_pool) - n_cs)
        n_cs = min(n_cs, len(english_pool) - n_en)

    # Partition source pools
    ta_for_mono = tamil_pool[:n_ta]
    ta_for_cs   = tamil_pool[n_ta: n_ta + n_cs]
    en_for_mono = english_pool[:n_en]
    en_for_cs   = english_pool[n_en: n_en + n_cs]

    # Build monolingual samples
    samples = []

    for seg in ta_for_mono:
        samples.append(_build_sample(
            seg["audio"], seg["transcript"], "monolingual_tamil"
        ))

    for seg in en_for_mono:
        samples.append(_build_sample(
            seg["audio"], seg["transcript"], "monolingual_english"
        ))

    # Build synthetic code-switched samples
    for ta_seg, en_seg in zip(ta_for_cs, en_for_cs):
        samples.append(_make_cs_sample(ta_seg, en_seg))

    logger.info(
        f"Dataset built — "
        f"monolingual_tamil: {n_ta}, "
        f"monolingual_english: {n_en}, "
        f"code_switched: {len([s for s in samples if s['segment_type']=='code_switched'])}"
    )
    return samples


def build_dataset_splits(
    samples: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """
    Stratified 80/10/10 split by segment_type.
    Returns {"train": [...], "validation": [...], "test": [...]}.
    """
    labels = [s["segment_type"] for s in samples]

    train_samples, temp = train_test_split(
        samples, test_size=(1 - train_ratio),
        stratify=labels, random_state=42,
    )
    temp_labels = [s["segment_type"] for s in temp]
    val_ratio_adj = val_ratio / (1 - train_ratio)

    val_samples, test_samples = train_test_split(
        temp, test_size=(1 - val_ratio_adj),
        stratify=temp_labels, random_state=42,
    )
    logger.info(
        f"Split — train: {len(train_samples)}, "
        f"val: {len(val_samples)}, test: {len(test_samples)}"
    )
    return {"train": train_samples, "validation": val_samples, "test": test_samples}


def print_dataset_stats(samples: list):
    from collections import Counter
    types  = Counter(s["segment_type"] for s in samples)
    total  = len(samples)
    print("\n=== Dataset Statistics ===")
    for seg_type, count in sorted(types.items()):
        print(f"  {seg_type}: {count} ({100*count/total:.1f}%)")
    print(f"  Total: {total} samples")
    cs = [s for s in samples if s["segment_type"] == "code_switched"]
    if cs:
        avg_dur = np.mean([s["duration_seconds"] for s in cs])
        print(f"  Avg code-switched duration: {avg_dur:.1f}s")
    print("==========================\n")


# ---------------------------------------------------------------------------
# Preserve original single-source loader for reference (not called by pipeline)
# ---------------------------------------------------------------------------

def preprocess_sample(sample: dict):
    """Kept for backwards compatibility. Not used by the main pipeline."""
    try:
        audio_data = sample.get("audio_filepath") or sample.get("audio") or {}
        audio_array = audio_data["array"]
        sr = audio_data["sampling_rate"]
        transcript = (
            sample.get("verbatim") or sample.get("normalized") or
            sample.get("text") or sample.get("transcript") or ""
        )
        if not transcript or len(transcript.strip()) < 3:
            return None
        audio = _resample(np.array(audio_array, dtype=np.float32), sr)
        duration = _duration(audio)
        if duration > MAX_AUDIO_S:
            audio = audio[:int(MAX_AUDIO_S * TARGET_SR)]
            duration = MAX_AUDIO_S
        return _build_sample(audio, transcript.strip(), tag_segment_type(transcript))
    except Exception as e:
        logger.warning(f"Skipping sample due to error: {e}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    authenticate_hf()

    samples = load_indicvoices_tamil()      # default: 200 samples
    print_dataset_stats(samples)

    splits = build_dataset_splits(samples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        save_path = DATA_DIR / f"{split_name}_metadata.json"
        metadata = [
            {k: v for k, v in s.items() if k != "audio"}
            for s in split_samples
        ]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {split_name} metadata → {save_path}")

    print("Dataset preparation complete.")

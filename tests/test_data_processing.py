"""
Tests for data/prepare_dataset.py

Tests only the pure processing functions — no HuggingFace downloads required.
"""

import numpy as np
import pytest
from data.prepare_dataset import (
    resample_audio,
    chunk_audio,
    tag_segment_type,
    count_switch_points,
    detect_language_mix,
    preprocess_sample,
)

TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# resample_audio
# ---------------------------------------------------------------------------

class TestResampleAudio:
    def test_already_correct_sr(self):
        audio = np.ones(TARGET_SR, dtype=np.float32)
        result = resample_audio(audio, TARGET_SR)
        assert len(result) == TARGET_SR
        assert result.dtype == np.float32

    def test_downsamples_from_44100(self):
        audio = np.random.randn(44100).astype(np.float32)
        result = resample_audio(audio, 44100)
        # resampled length should be ~16000 (within 5%)
        assert abs(len(result) - TARGET_SR) < TARGET_SR * 0.05

    def test_stereo_collapsed_to_mono(self):
        # shape (2, N) — two channels
        audio = np.random.randn(2, TARGET_SR).astype(np.float32)
        result = resample_audio(audio, TARGET_SR)
        assert result.ndim == 1

    def test_output_dtype_is_float32(self):
        audio = np.ones(TARGET_SR, dtype=np.float64)
        result = resample_audio(audio, TARGET_SR)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# chunk_audio
# ---------------------------------------------------------------------------

class TestChunkAudio:
    def test_short_audio_not_chunked(self):
        audio = np.zeros(TARGET_SR * 10)  # 10 seconds
        chunks = chunk_audio(audio)
        assert len(chunks) == 1

    def test_exactly_30s_not_chunked(self):
        audio = np.zeros(TARGET_SR * 30)
        chunks = chunk_audio(audio)
        assert len(chunks) == 1

    def test_long_audio_split(self):
        audio = np.zeros(TARGET_SR * 65)  # 65 seconds
        chunks = chunk_audio(audio)
        assert len(chunks) >= 2

    def test_chunks_under_30s(self):
        audio = np.zeros(TARGET_SR * 65)
        chunks = chunk_audio(audio)
        for chunk in chunks:
            assert len(chunk) <= TARGET_SR * 30

    def test_sub_1s_chunk_dropped(self):
        # 30s + 0.5s → only first chunk kept
        audio = np.zeros(TARGET_SR * 30 + TARGET_SR // 2)
        chunks = chunk_audio(audio)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# tag_segment_type
# ---------------------------------------------------------------------------

class TestTagSegmentType:
    def test_valid_output_values(self):
        valid = {"monolingual_tamil", "monolingual_english", "code_switched"}
        result = tag_segment_type("hello world how are you")
        assert result in valid

    def test_english_text_tagged_english(self):
        result = tag_segment_type(
            "the quick brown fox jumps over the lazy dog"
        )
        assert result == "monolingual_english"

    def test_returns_string(self):
        result = tag_segment_type("test")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# count_switch_points
# ---------------------------------------------------------------------------

class TestCountSwitchPoints:
    def test_single_word_no_switches(self):
        assert count_switch_points("hello") == 0

    def test_returns_nonnegative_int(self):
        result = count_switch_points("hello world vanakkam")
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_string(self):
        assert count_switch_points("") == 0


# ---------------------------------------------------------------------------
# detect_language_mix
# ---------------------------------------------------------------------------

class TestDetectLanguageMix:
    def test_returns_dict(self):
        result = detect_language_mix("hello world")
        assert isinstance(result, dict)

    def test_probabilities_sum_to_one(self):
        result = detect_language_mix("hello world how are you doing today")
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_all_values_between_0_and_1(self):
        result = detect_language_mix("hello world")
        for prob in result.values():
            assert 0.0 <= prob <= 1.0

    def test_handles_empty_string(self):
        result = detect_language_mix("")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# preprocess_sample
# ---------------------------------------------------------------------------

class TestPreprocessSample:
    def _make_sample(self, transcript="hello world", duration_s=3):
        audio = np.zeros(TARGET_SR * duration_s, dtype=np.float32)
        return {
            "audio": {"array": audio, "sampling_rate": TARGET_SR},
            "text": transcript,
        }

    def test_valid_sample_returns_dict(self):
        result = preprocess_sample(self._make_sample())
        assert isinstance(result, dict)

    def test_output_keys_present(self):
        result = preprocess_sample(self._make_sample())
        assert result is not None
        expected_keys = {
            "audio", "transcript", "segment_type", "switch_count",
            "lang_mix_en", "lang_mix_ta", "duration_seconds", "sample_rate",
        }
        assert expected_keys.issubset(result.keys())

    def test_short_transcript_skipped(self):
        sample = self._make_sample(transcript="hi")
        result = preprocess_sample(sample)
        assert result is None

    def test_empty_transcript_skipped(self):
        sample = self._make_sample(transcript="")
        result = preprocess_sample(sample)
        assert result is None

    def test_long_audio_truncated_to_30s(self):
        sample = self._make_sample(
            transcript="this is a long recording", duration_s=60
        )
        result = preprocess_sample(sample)
        assert result is not None
        assert result["duration_seconds"] <= 30.0

    def test_sample_rate_is_16000(self):
        result = preprocess_sample(self._make_sample())
        assert result is not None
        assert result["sample_rate"] == TARGET_SR

    def test_uses_transcript_key_as_fallback(self):
        audio = np.zeros(TARGET_SR * 3, dtype=np.float32)
        sample = {
            "audio": {"array": audio, "sampling_rate": TARGET_SR},
            "transcript": "hello world this is a test",
        }
        result = preprocess_sample(sample)
        assert result is not None
        assert result["transcript"] == "hello world this is a test"

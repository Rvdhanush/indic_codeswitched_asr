"""
Tests for data/prepare_dataset.py

Tests only the pure processing functions — no HuggingFace downloads required.
"""

import numpy as np
import pytest
from data.prepare_dataset import (
    _build_sample,
    _make_cs_sample,
    _resample,
    _trim_to_window,
    build_provenance,
    count_switch_points,
    detect_language_mix,
    preprocess_sample,
    resolve_dataset_revision,
    tag_segment_type,
)

TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# _resample
# ---------------------------------------------------------------------------

class TestResample:
    def test_already_correct_sr(self):
        audio = np.ones(TARGET_SR, dtype=np.float32)
        result = _resample(audio, TARGET_SR)
        assert len(result) == TARGET_SR
        assert result.dtype == np.float32

    def test_downsamples_from_44100(self):
        audio = np.random.randn(44100).astype(np.float32)
        result = _resample(audio, 44100)
        # resampled length should be ~16000 (within 5%)
        assert abs(len(result) - TARGET_SR) < TARGET_SR * 0.05

    def test_stereo_collapsed_to_mono(self):
        # shape (2, N) — two channels
        audio = np.random.randn(2, TARGET_SR).astype(np.float32)
        result = _resample(audio, TARGET_SR)
        assert result.ndim == 1

    def test_output_dtype_is_float32(self):
        audio = np.ones(TARGET_SR, dtype=np.float64)
        result = _resample(audio, TARGET_SR)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _trim_to_window
# ---------------------------------------------------------------------------

class TestTrimToWindow:
    def test_short_audio_unchanged(self):
        audio = np.zeros(TARGET_SR * 5, dtype=np.float32)  # 5s
        result = _trim_to_window(audio, max_s=8.0)
        assert len(result) == len(audio)

    def test_exactly_at_limit_unchanged(self):
        audio = np.zeros(TARGET_SR * 8, dtype=np.float32)  # 8s
        result = _trim_to_window(audio, max_s=8.0)
        assert len(result) == TARGET_SR * 8

    def test_long_audio_trimmed(self):
        audio = np.zeros(TARGET_SR * 20, dtype=np.float32)  # 20s
        result = _trim_to_window(audio, max_s=8.0)
        assert len(result) <= TARGET_SR * 8

    def test_result_never_exceeds_limit(self):
        audio = np.zeros(TARGET_SR * 65, dtype=np.float32)
        result = _trim_to_window(audio, max_s=30.0)
        assert len(result) <= TARGET_SR * 30

    def test_custom_max_s_respected(self):
        audio = np.zeros(TARGET_SR * 10, dtype=np.float32)
        result = _trim_to_window(audio, max_s=3.0)
        assert len(result) == TARGET_SR * 3


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


# ---------------------------------------------------------------------------
# Provenance
#
# The frozen corpus is only reproducible if the manifest records what produced
# it. These check the record is complete, not that a rebuild succeeds.
# ---------------------------------------------------------------------------

class TestBuildProvenance:
    def test_records_both_source_revisions(self):
        prov = build_provenance(tamil_revision="aaa111", english_revision="bbb222")
        assert prov["sources"]["tamil"]["revision"] == "aaa111"
        assert prov["sources"]["english"]["revision"] == "bbb222"

    def test_names_both_datasets(self):
        prov = build_provenance()
        assert prov["sources"]["tamil"]["dataset"] == "SPRINGLab/IndicVoices-R_Tamil"
        assert prov["sources"]["english"]["dataset"] == "librispeech_asr"

    def test_unpinned_revisions_are_explicit_nulls(self):
        prov = build_provenance()
        assert prov["sources"]["tamil"]["revision"] is None

    def test_records_the_builder_constants(self):
        prov = build_provenance(size=1500)
        builder = prov["builder"]
        assert builder["requested_size"] == 1500
        for key in ("min_duration_s", "max_duration_s", "silence_s", "target_sr",
                    "n_code_switched", "n_mono_tamil", "n_mono_english"):
            assert key in builder

    def test_records_the_split_parameters(self):
        prov = build_provenance(train_ratio=0.8, val_ratio=0.1)
        assert prov["split"] == {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "random_state": 42,
            "stratify_by": "segment_type",
        }

    def test_is_json_serialisable(self):
        import json
        json.dumps(build_provenance(size=10))


class TestResolveDatasetRevision:
    def test_returns_none_for_a_nonexistent_repo(self):
        """Offline or missing repo must degrade, not raise, so the build can
        proceed unpinned and say so."""
        assert resolve_dataset_revision("this-org/does-not-exist-xyz") is None


# ---------------------------------------------------------------------------
# Per-sample source provenance
# ---------------------------------------------------------------------------

class TestSampleSource:
    def test_build_sample_carries_source(self):
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        src = {"dataset": "d", "revision": "r", "stream_index": 7}
        sample = _build_sample(audio, "hello world", "monolingual_english", source=src)
        assert sample["source"] == src

    def test_source_defaults_to_none(self):
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        sample = _build_sample(audio, "hello world", "monolingual_english")
        assert sample["source"] is None

    def test_cs_sample_records_both_halves(self):
        seg_ta = {
            "audio": np.zeros(TARGET_SR, dtype=np.float32),
            "transcript": "tamil text",
            "source": {"dataset": "ta", "revision": "r1", "stream_index": 1},
        }
        seg_en = {
            "audio": np.zeros(TARGET_SR, dtype=np.float32),
            "transcript": "english text",
            "source": {"dataset": "en", "revision": "r2", "stream_index": 2},
        }
        sample = _make_cs_sample(seg_ta, seg_en)
        assert sample["source"]["tamil"]["dataset"] == "ta"
        assert sample["source"]["english"]["dataset"] == "en"
        assert sample["segment_type"] == "code_switched"

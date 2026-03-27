"""
Tests for evaluation/metrics.py

All tests are pure Python — no models, no GPU, no HuggingFace required.
"""

import pytest
from evaluation.metrics import (
    compute_wer,
    compute_cer,
    categorize_failure,
    analyze_failures,
    compute_stratified_wer,
)


# ---------------------------------------------------------------------------
# compute_wer
# ---------------------------------------------------------------------------

class TestComputeWer:
    def test_perfect_match(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_full_substitution(self):
        # one word replaced → 1/1 = 1.0
        assert compute_wer("hello", "world") == 1.0

    def test_case_insensitive(self):
        assert compute_wer("Hello World", "hello world") == 0.0

    def test_punctuation_ignored(self):
        assert compute_wer("hello, world!", "hello world") == 0.0

    def test_partial_error(self):
        wer = compute_wer("one two three four", "one two three five")
        assert 0 < wer < 1.0

    def test_empty_hypothesis(self):
        # all words deleted
        wer = compute_wer("hello world", "")
        assert wer == 1.0


# ---------------------------------------------------------------------------
# compute_cer
# ---------------------------------------------------------------------------

class TestComputeCer:
    def test_perfect_match(self):
        assert compute_cer("abc", "abc") == 0.0

    def test_nonzero_on_diff(self):
        assert compute_cer("abc", "xyz") > 0.0


# ---------------------------------------------------------------------------
# categorize_failure
# ---------------------------------------------------------------------------

class TestCategorizeFailure:
    def test_insertion_filler(self):
        ref = ["hello", "world"]
        hyp = ["hello", "um", "world"]
        assert categorize_failure(ref, hyp, []) == "INSERTION_FILLER"

    def test_substitution_number(self):
        ref = ["call", "100", "times"]
        hyp = ["call", "times"]
        assert categorize_failure(ref, hyp, []) == "SUBSTITUTION_NUMBER"

    def test_deletion_proper_noun(self):
        ref = ["visit", "Chennai", "today"]
        hyp = ["visit", "today"]
        assert categorize_failure(ref, hyp, []) == "DELETION_PROPER_NOUN"

    def test_substitution_switch_near_boundary(self):
        # boundary at index 2; ref[2] != hyp[2]
        ref = ["i", "am", "going", "naalai"]
        hyp = ["i", "am", "coming", "naalai"]
        boundaries = [2]
        assert categorize_failure(ref, hyp, boundaries) == "SUBSTITUTION_SWITCH"

    def test_language_confusion_default(self):
        # no filler, no number, no proper noun, no switch boundary
        ref = ["vanakkam", "friends"]
        hyp = ["hello", "friends"]
        assert categorize_failure(ref, hyp, []) == "LANGUAGE_CONFUSION"

    def test_filler_takes_priority_over_proper_noun(self):
        # both filler and proper noun present — filler checked first
        ref = ["Visit", "Chennai"]
        hyp = ["Visit", "um", "somewhere"]
        assert categorize_failure(ref, hyp, []) == "INSERTION_FILLER"


# ---------------------------------------------------------------------------
# analyze_failures
# ---------------------------------------------------------------------------

class TestAnalyzeFailures:
    def test_returns_required_keys(self):
        result = analyze_failures("hello world", "hello world")
        required = {
            "wer", "cer", "segment_type", "switch_boundary_count",
            "failure_type", "reference_word_count", "hypothesis_word_count",
        }
        assert required.issubset(result.keys())

    def test_wer_range(self):
        result = analyze_failures("one two three", "one two four")
        assert 0.0 <= result["wer"] <= 1.0

    def test_word_counts(self):
        result = analyze_failures("one two three", "one two")
        assert result["reference_word_count"] == 3
        assert result["hypothesis_word_count"] == 2

    def test_segment_type_is_valid(self):
        result = analyze_failures("hello world", "hello world")
        assert result["segment_type"] in (
            "monolingual_tamil", "monolingual_english", "code_switched"
        )

    def test_failure_type_is_valid(self):
        valid = {
            "SUBSTITUTION_SWITCH", "DELETION_PROPER_NOUN",
            "SUBSTITUTION_NUMBER", "LANGUAGE_CONFUSION", "INSERTION_FILLER",
        }
        result = analyze_failures("hello world", "goodbye moon")
        assert result["failure_type"] in valid


# ---------------------------------------------------------------------------
# compute_stratified_wer
# ---------------------------------------------------------------------------

class TestComputeStratifiedWer:
    def _make_result(self, wer, seg_type, failure_type="LANGUAGE_CONFUSION"):
        return {
            "wer": wer,
            "segment_type": seg_type,
            "failure_type": failure_type,
        }

    def test_overall_wer_average(self):
        results = [
            self._make_result(0.2, "monolingual_tamil"),
            self._make_result(0.4, "monolingual_tamil"),
        ]
        out = compute_stratified_wer(results)
        assert out["overall_wer"] == pytest.approx(0.3, abs=1e-4)

    def test_segment_type_isolation(self):
        results = [
            self._make_result(0.1, "monolingual_tamil"),
            self._make_result(0.5, "code_switched"),
        ]
        out = compute_stratified_wer(results)
        assert out["monolingual_tamil_wer"] == pytest.approx(0.1, abs=1e-4)
        assert out["code_switched_wer"] == pytest.approx(0.5, abs=1e-4)
        assert out["monolingual_english_wer"] is None  # no English samples

    def test_failure_breakdown_counts(self):
        results = [
            self._make_result(0.3, "code_switched", "LANGUAGE_CONFUSION"),
            self._make_result(0.5, "code_switched", "LANGUAGE_CONFUSION"),
            self._make_result(0.2, "monolingual_tamil", "INSERTION_FILLER"),
        ]
        out = compute_stratified_wer(results)
        assert out["failure_breakdown"]["LANGUAGE_CONFUSION"] == 2
        assert out["failure_breakdown"]["INSERTION_FILLER"] == 1

    def test_total_samples(self):
        results = [self._make_result(0.1, "code_switched") for _ in range(5)]
        out = compute_stratified_wer(results)
        assert out["total_samples"] == 5

    def test_empty_input(self):
        out = compute_stratified_wer([])
        assert out["overall_wer"] is None
        assert out["total_samples"] == 0

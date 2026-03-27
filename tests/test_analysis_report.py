"""
Tests for analysis/report.py

Uses fixture data — no results files or models required.
"""

import json
import pytest
from pathlib import Path
from analysis.report import (
    code_switch_penalty,
    dominant_failure,
    shared_failures,
    wer_ranking,
    build_markdown,
    build_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    model_key="test_model",
    model_name="test/model",
    overall_wer=0.35,
    mono_ta=0.20,
    mono_en=0.15,
    cs_wer=0.60,
    breakdown=None,
):
    if breakdown is None:
        breakdown = {
            "SUBSTITUTION_SWITCH": 10,
            "DELETION_PROPER_NOUN": 5,
            "SUBSTITUTION_NUMBER": 3,
            "LANGUAGE_CONFUSION": 20,
            "INSERTION_FILLER": 2,
        }
    return {
        "model_name": model_name,
        "model_key": model_key,
        "overall_wer": overall_wer,
        "monolingual_tamil_wer": mono_ta,
        "monolingual_english_wer": mono_en,
        "code_switched_wer": cs_wer,
        "failure_breakdown": breakdown,
        "total_samples": 100,
        "errors": 0,
        "device": "cpu",
    }


@pytest.fixture
def single_result():
    return _make_result()


@pytest.fixture
def three_results():
    return {
        "whisper_medium": _make_result(
            "whisper_medium", overall_wer=0.40, cs_wer=0.70,
            breakdown={
                "SUBSTITUTION_SWITCH": 15,
                "DELETION_PROPER_NOUN": 3,
                "SUBSTITUTION_NUMBER": 2,
                "LANGUAGE_CONFUSION": 5,
                "INSERTION_FILLER": 1,
            }
        ),
        "indic_whisper": _make_result(
            "indic_whisper", overall_wer=0.35, cs_wer=0.55,
            breakdown={
                "SUBSTITUTION_SWITCH": 12,
                "DELETION_PROPER_NOUN": 6,
                "SUBSTITUTION_NUMBER": 4,
                "LANGUAGE_CONFUSION": 3,
                "INSERTION_FILLER": 2,
            }
        ),
        "indicwav2vec": _make_result(
            "indicwav2vec", overall_wer=0.50, cs_wer=0.80,
            breakdown={
                "SUBSTITUTION_SWITCH": 18,
                "DELETION_PROPER_NOUN": 4,
                "SUBSTITUTION_NUMBER": 5,
                "LANGUAGE_CONFUSION": 2,
                "INSERTION_FILLER": 1,
            }
        ),
    }


# ---------------------------------------------------------------------------
# code_switch_penalty
# ---------------------------------------------------------------------------

class TestCodeSwitchPenalty:
    def test_basic_penalty(self, single_result):
        # cs_wer=0.60, avg_mono=(0.20+0.15)/2=0.175 → 0.60/0.175 ≈ 3.43
        penalty = code_switch_penalty(single_result)
        assert penalty == pytest.approx(0.60 / 0.175, rel=1e-2)

    def test_no_penalty_when_equal(self):
        r = _make_result(mono_ta=0.40, mono_en=0.40, cs_wer=0.40)
        assert code_switch_penalty(r) == pytest.approx(1.0, rel=1e-2)

    def test_returns_none_when_cs_missing(self):
        r = _make_result()
        r["code_switched_wer"] = None
        assert code_switch_penalty(r) is None

    def test_returns_none_when_both_mono_missing(self):
        r = _make_result()
        r["monolingual_tamil_wer"] = None
        r["monolingual_english_wer"] = None
        assert code_switch_penalty(r) is None

    def test_uses_only_available_mono(self):
        r = _make_result(mono_en=None, mono_ta=0.20, cs_wer=0.60)
        penalty = code_switch_penalty(r)
        assert penalty == pytest.approx(3.0, rel=1e-2)

    def test_returns_none_when_avg_mono_is_zero(self):
        r = _make_result(mono_ta=0.0, mono_en=0.0, cs_wer=0.5)
        assert code_switch_penalty(r) is None


# ---------------------------------------------------------------------------
# dominant_failure
# ---------------------------------------------------------------------------

class TestDominantFailure:
    def test_identifies_dominant_category(self):
        breakdown = {
            "SUBSTITUTION_SWITCH": 5,
            "DELETION_PROPER_NOUN": 2,
            "SUBSTITUTION_NUMBER": 1,
            "LANGUAGE_CONFUSION": 20,
            "INSERTION_FILLER": 1,
        }
        cat, share = dominant_failure(breakdown)
        assert cat == "LANGUAGE_CONFUSION"
        assert share == pytest.approx(20 / 29 * 100, rel=1e-1)

    def test_empty_breakdown(self):
        cat, share = dominant_failure({})
        assert cat == "N/A"
        assert share == 0.0

    def test_all_zero_counts(self):
        breakdown = {k: 0 for k in [
            "SUBSTITUTION_SWITCH", "DELETION_PROPER_NOUN",
            "SUBSTITUTION_NUMBER", "LANGUAGE_CONFUSION", "INSERTION_FILLER"
        ]}
        cat, share = dominant_failure(breakdown)
        assert cat == "N/A"


# ---------------------------------------------------------------------------
# shared_failures
# ---------------------------------------------------------------------------

class TestSharedFailures:
    def test_finds_common_top2(self, three_results):
        # All three models have SUBSTITUTION_SWITCH as top-1
        shared = shared_failures(three_results)
        assert "SUBSTITUTION_SWITCH" in shared

    def test_empty_results(self):
        assert shared_failures({}) == []

    def test_single_model_returns_its_top2(self):
        results = {"m": _make_result(breakdown={
            "SUBSTITUTION_SWITCH": 10,
            "LANGUAGE_CONFUSION": 8,
            "DELETION_PROPER_NOUN": 2,
            "SUBSTITUTION_NUMBER": 1,
            "INSERTION_FILLER": 0,
        })}
        shared = shared_failures(results)
        assert set(shared) == {"SUBSTITUTION_SWITCH", "LANGUAGE_CONFUSION"}


# ---------------------------------------------------------------------------
# wer_ranking
# ---------------------------------------------------------------------------

class TestWerRanking:
    def test_sorted_ascending(self, three_results):
        ranking = wer_ranking(three_results, "overall_wer")
        wers = [wer for _, wer in ranking]
        assert wers == sorted(wers)

    def test_best_model_first(self, three_results):
        ranking = wer_ranking(three_results, "code_switched_wer")
        assert ranking[0][0] == "indic_whisper"  # cs_wer=0.55

    def test_skips_none_values(self):
        results = {
            "a": _make_result(overall_wer=0.3),
            "b": {**_make_result(), "overall_wer": None},
        }
        ranking = wer_ranking(results, "overall_wer")
        assert len(ranking) == 1
        assert ranking[0][0] == "a"


# ---------------------------------------------------------------------------
# build_markdown
# ---------------------------------------------------------------------------

class TestBuildMarkdown:
    def test_returns_string(self, three_results):
        md = build_markdown(three_results)
        assert isinstance(md, str)

    def test_contains_required_sections(self, three_results):
        md = build_markdown(three_results)
        assert "## 1." in md
        assert "## 2." in md
        assert "## 3." in md
        assert "## 4." in md

    def test_contains_all_failure_categories(self, three_results):
        md = build_markdown(three_results)
        for cat in [
            "SUBSTITUTION_SWITCH", "DELETION_PROPER_NOUN",
            "SUBSTITUTION_NUMBER", "LANGUAGE_CONFUSION", "INSERTION_FILLER",
        ]:
            assert cat in md

    def test_contains_model_names(self, three_results):
        md = build_markdown(three_results)
        assert "Whisper-medium" in md
        assert "IndicWhisper" in md
        assert "IndicWav2Vec" in md

    def test_cs_penalty_present(self, three_results):
        md = build_markdown(three_results)
        assert "CS Penalty" in md


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:
    def test_returns_dict(self, three_results):
        summary = build_summary(three_results)
        assert isinstance(summary, dict)

    def test_has_meta_key(self, three_results):
        summary = build_summary(three_results)
        assert "_meta" in summary

    def test_meta_contains_rankings(self, three_results):
        summary = build_summary(three_results)
        assert "cs_wer_ranking" in summary["_meta"]
        assert "overall_wer_ranking" in summary["_meta"]

    def test_each_model_has_penalty(self, three_results):
        summary = build_summary(three_results)
        for key in three_results:
            assert "code_switch_penalty" in summary[key]

    def test_dominant_failure_populated(self, three_results):
        summary = build_summary(three_results)
        for key in three_results:
            assert summary[key]["dominant_failure_category"] != ""

    def test_serialisable_to_json(self, three_results):
        summary = build_summary(three_results)
        dumped = json.dumps(summary)
        assert isinstance(dumped, str)


# ---------------------------------------------------------------------------
# Integration: build_markdown + build_summary round-trip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_report_generated_from_summary(self, three_results):
        summary = build_summary(three_results)
        # summary (minus _meta) should be usable as input to build_markdown
        without_meta = {k: v for k, v in summary.items() if k != "_meta"}
        md = build_markdown(without_meta)
        assert len(md) > 100

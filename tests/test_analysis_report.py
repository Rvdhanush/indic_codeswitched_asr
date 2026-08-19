"""
Tests for analysis/report.py

Uses fixture data — no results files or models required.
"""

import json
import pytest
from pathlib import Path
from analysis.report import (
    code_switch_penalty,
    corpus_consistency,
    dominant_failure,
    shared_failures,
    wer_ranking,
    build_markdown,
    build_summary,
    main as report_main,
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
    corpus_id="c0ffeec0ffee0001",
    split="test",
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
        "corpus_id": corpus_id,
        "split": split,
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
        "whisper_small": _make_result(
            "whisper_small", overall_wer=0.40, cs_wer=0.70,
            breakdown={
                "SUBSTITUTION_SWITCH": 15,
                "DELETION_PROPER_NOUN": 3,
                "SUBSTITUTION_NUMBER": 2,
                "LANGUAGE_CONFUSION": 5,
                "INSERTION_FILLER": 1,
            }
        ),
        "whisper_tamil": _make_result(
            "whisper_tamil", overall_wer=0.35, cs_wer=0.55,
            breakdown={
                "SUBSTITUTION_SWITCH": 12,
                "DELETION_PROPER_NOUN": 6,
                "SUBSTITUTION_NUMBER": 4,
                "LANGUAGE_CONFUSION": 3,
                "INSERTION_FILLER": 2,
            }
        ),
        "wav2vec2_tamil": _make_result(
            "wav2vec2_tamil", overall_wer=0.50, cs_wer=0.80,
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
        assert ranking[0][0] == "whisper_tamil"  # cs_wer=0.55

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
        assert "Whisper-small (baseline)" in md
        assert "Whisper-tamil-medium" in md
        assert "Wav2Vec2-tamil" in md

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


# ---------------------------------------------------------------------------
# corpus comparability
#
# These cover the guard that would have caught the retracted result: a table
# built from models scored on different test sets must not render as a clean
# comparison, and must not exit 0.
# ---------------------------------------------------------------------------

class TestCorpusConsistency:
    def test_matching_ids_are_consistent(self, three_results):
        status, ids = corpus_consistency(three_results)
        assert status == "consistent"
        assert set(ids.values()) == {"c0ffeec0ffee0001"}

    def test_differing_ids_are_mismatched(self, three_results):
        three_results["wav2vec2_tamil"]["corpus_id"] = "beefbeefbeef0002"
        status, ids = corpus_consistency(three_results)
        assert status == "mismatched"
        assert ids["wav2vec2_tamil"] == "beefbeefbeef0002"

    def test_missing_id_is_unknown(self, three_results):
        del three_results["whisper_small"]["corpus_id"]
        status, _ = corpus_consistency(three_results)
        assert status == "unknown"

    def test_null_id_is_unknown(self, three_results):
        three_results["whisper_small"]["corpus_id"] = None
        assert corpus_consistency(three_results)[0] == "unknown"

    def test_empty_results_are_unknown(self):
        assert corpus_consistency({})[0] == "unknown"


class TestBanner:
    def test_consistent_run_reports_provenance(self, three_results):
        md = build_markdown(three_results)
        assert "c0ffeec0ffee0001" in md
        assert "Every row below was scored on the same samples." in md
        assert "not currently valid" not in md

    def test_consistent_run_keeps_metric_caveats(self, three_results):
        md = build_markdown(three_results)
        assert "macro-averaged per utterance" in md

    def test_mismatch_refuses_to_present_a_comparison(self, three_results):
        three_results["wav2vec2_tamil"]["corpus_id"] = "beefbeefbeef0002"
        md = build_markdown(three_results)
        assert "scored on different data" in md
        assert "beefbeefbeef0002" in md

    def test_unstamped_results_keep_the_invalidity_banner(self, three_results):
        for r in three_results.values():
            r.pop("corpus_id")
        md = build_markdown(three_results)
        assert "not currently valid" in md


class TestSummaryProvenance:
    def test_records_verdict(self, three_results):
        summary = build_summary(three_results)
        assert summary["_meta"]["corpus_status"] == "consistent"
        assert summary["_meta"]["comparable"] is True

    def test_records_mismatch(self, three_results):
        three_results["wav2vec2_tamil"]["corpus_id"] = "beefbeefbeef0002"
        summary = build_summary(three_results)
        assert summary["_meta"]["comparable"] is False
        assert summary["_meta"]["corpus_ids"]["wav2vec2_tamil"] == "beefbeefbeef0002"

    def test_carries_per_model_corpus_id(self, three_results):
        summary = build_summary(three_results)
        assert summary["whisper_small"]["corpus_id"] == "c0ffeec0ffee0001"
        assert summary["whisper_small"]["split"] == "test"


class TestReportExitCode:
    def _write(self, tmp_path, results):
        path = tmp_path / "baseline_wer_all.json"
        path.write_text(json.dumps(results), encoding="utf-8")
        return path

    def test_exits_zero_when_comparable(self, tmp_path, three_results, monkeypatch):
        path = self._write(tmp_path, three_results)
        monkeypatch.setattr(
            "sys.argv",
            ["report", "--results", str(path), "--out-dir", str(tmp_path)],
        )
        assert report_main() == 0

    def test_exits_nonzero_on_mismatch(self, tmp_path, three_results, monkeypatch):
        three_results["wav2vec2_tamil"]["corpus_id"] = "beefbeefbeef0002"
        path = self._write(tmp_path, three_results)
        monkeypatch.setattr(
            "sys.argv",
            ["report", "--results", str(path), "--out-dir", str(tmp_path)],
        )
        assert report_main() == 1
        # The report is still written, so the banner is readable.
        assert (tmp_path / "failure_analysis_report.md").exists()

    def test_exits_nonzero_on_unstamped_results(self, tmp_path, three_results, monkeypatch):
        for r in three_results.values():
            r.pop("corpus_id")
        path = self._write(tmp_path, three_results)
        monkeypatch.setattr(
            "sys.argv",
            ["report", "--results", str(path), "--out-dir", str(tmp_path)],
        )
        assert report_main() == 1

    def test_exits_nonzero_when_results_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "report",
                "--results", str(tmp_path / "nope.json"),
                "--out-dir", str(tmp_path),
            ],
        )
        assert report_main() == 1

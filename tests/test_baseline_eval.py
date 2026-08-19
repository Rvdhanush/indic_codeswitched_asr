"""
Tests for evaluation/baseline_eval.py

Focused on the guards around results files, not on model quality: no model is
loaded here. `evaluate_model` is monkeypatched, so these run offline and fast.

The defect these exist to prevent: a run that scores models on different data,
or that fails partway, must never leave results/baseline_wer_all.json looking
like a valid comparison.
"""

import json

import pytest

from evaluation import baseline_eval as B


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _result(model_key, corpus_id="c0ffeec0ffee0001"):
    return {
        "model_name": f"org/{model_key}",
        "model_key": model_key,
        "device": "cpu",
        "split": "test",
        "corpus_id": corpus_id,
        "split_corpus_id": corpus_id,
        "total_samples": 10,
        "errors": 0,
        "overall_wer": 0.5,
        "monolingual_tamil_wer": 0.5,
        "monolingual_english_wer": 0.5,
        "code_switched_wer": 0.5,
        "failure_breakdown": {"SUBSTITUTION_SWITCH": 1},
    }


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "RESULTS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def stub_eval(monkeypatch):
    """Replace evaluate_model with a stub whose corpus_id is controllable."""
    ids = {}

    def _install(corpus_ids=None):
        ids.update(corpus_ids or {})

        def fake(model_key, model_config, test_samples, max_samples=None):
            return _result(model_key, ids.get(model_key, "c0ffeec0ffee0001"))

        monkeypatch.setattr(B, "evaluate_model", fake)

    return _install


# ---------------------------------------------------------------------------
# Model key validation
# ---------------------------------------------------------------------------

class TestModelKeys:
    def test_unknown_key_raises(self, results_dir, stub_eval):
        stub_eval()
        with pytest.raises(ValueError, match="Unknown model key"):
            B.run_all_baselines([], models_to_run=["whisper_medium"])

    def test_unknown_key_does_not_write_anything(self, results_dir, stub_eval):
        stub_eval()
        with pytest.raises(ValueError):
            B.run_all_baselines([], models_to_run=["nope"])
        assert list(results_dir.glob("*.json")) == []

    def test_known_keys_accepted(self, results_dir, stub_eval):
        stub_eval()
        out = B.run_all_baselines([], models_to_run=["whisper_small"])
        assert set(out) == {"whisper_small"}


# ---------------------------------------------------------------------------
# The combined comparison file
# ---------------------------------------------------------------------------

class TestCombinedFile:
    def test_written_when_all_models_share_a_corpus(self, results_dir, stub_eval):
        stub_eval()
        B.run_all_baselines([], models_to_run=list(B.MODELS))
        combined = results_dir / "baseline_wer_all.json"
        assert combined.exists()
        assert set(json.loads(combined.read_text())) == set(B.MODELS)

    def test_not_written_when_corpora_differ(self, results_dir, stub_eval):
        stub_eval({"wav2vec2_tamil": "beefbeefbeef0002"})
        B.run_all_baselines([], models_to_run=list(B.MODELS))
        assert not (results_dir / "baseline_wer_all.json").exists()

    def test_existing_combined_file_survives_a_mismatch(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        previous = {"whisper_small_lora": _result("whisper_small_lora")}
        combined.write_text(json.dumps(previous), encoding="utf-8")
        stub_eval({"wav2vec2_tamil": "beefbeefbeef0002"})
        B.run_all_baselines([], models_to_run=list(B.MODELS))
        assert json.loads(combined.read_text()) == previous

    def test_per_model_files_are_still_written_on_mismatch(self, results_dir, stub_eval):
        stub_eval({"wav2vec2_tamil": "beefbeefbeef0002"})
        B.run_all_baselines([], models_to_run=list(B.MODELS))
        for key in B.MODELS:
            assert (results_dir / f"{key}_wer.json").exists()

    def test_partial_failure_leaves_combined_untouched(self, results_dir, monkeypatch):
        combined = results_dir / "baseline_wer_all.json"
        combined.write_text(json.dumps({"previous": "results"}), encoding="utf-8")

        def flaky(model_key, model_config, test_samples, max_samples=None):
            if model_key == "wav2vec2_tamil":
                raise RuntimeError("out of memory")
            return _result(model_key)

        monkeypatch.setattr(B, "evaluate_model", flaky)
        out = B.run_all_baselines([], models_to_run=list(B.MODELS))
        assert "wav2vec2_tamil" not in out
        assert json.loads(combined.read_text()) == {"previous": "results"}

    def test_total_failure_leaves_combined_untouched(self, results_dir, monkeypatch):
        combined = results_dir / "baseline_wer_all.json"
        combined.write_text(json.dumps({"previous": "results"}), encoding="utf-8")

        def always_fails(*a, **kw):
            raise RuntimeError("nope")

        monkeypatch.setattr(B, "evaluate_model", always_fails)
        assert B.run_all_baselines([], models_to_run=list(B.MODELS)) == {}
        assert json.loads(combined.read_text()) == {"previous": "results"}

    def test_results_carry_corpus_identity(self, results_dir, stub_eval):
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        written = json.loads((results_dir / "whisper_small_wer.json").read_text())
        assert written["corpus_id"] == "c0ffeec0ffee0001"
        assert written["split"] == "test"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_dry_run_writes_nothing(self, results_dir, capsys):
        assert B.main(["--dry-run"]) == 0
        assert list(results_dir.glob("*.json")) == []

    def test_dry_run_reports_the_corpus(self, tmp_path, capsys):
        import numpy as np
        from data import corpus as C

        sample = {
            "audio": np.zeros(1600, dtype=np.float32),
            "transcript": "hello",
            "segment_type": "monolingual_english",
            "switch_count": 0,
            "sample_rate": 16000,
        }
        manifest = C.freeze({"test": [sample]}, corpus_dir=tmp_path)
        B.main(["--dry-run", "--corpus-dir", str(tmp_path)])
        assert manifest["split_corpus_ids"]["test"] in capsys.readouterr().out

    def test_unknown_model_key_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            B.main(["--models", "whisper_medium", "--dry-run"])

    def test_missing_corpus_exits_nonzero(self, tmp_path, results_dir):
        assert B.main(["--models", "whisper_small",
                       "--corpus-dir", str(tmp_path / "nothing")]) == 1

    def test_missing_corpus_writes_nothing(self, tmp_path, results_dir):
        B.main(["--models", "whisper_small", "--corpus-dir", str(tmp_path / "nothing")])
        assert list(results_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Corpus stamping inside evaluate_model
#
# The stamp must describe the samples that actually contributed to the score,
# not the split they were drawn from. A cap or a failed sample changes it.
# ---------------------------------------------------------------------------

def _corpus_sample(uid, transcript="hello world"):
    import numpy as np
    return {
        "uid": uid,
        "split": "test",
        "corpus_id": "sp1itc0rpus00001",
        "audio": np.zeros(1600, dtype=np.float32),
        "transcript": transcript,
        "segment_type": "monolingual_english",
        "switch_count": 0,
    }


@pytest.fixture
def stub_whisper(monkeypatch):
    monkeypatch.setattr(B, "load_whisper_model", lambda name: (object(), object()))
    monkeypatch.setattr(
        B, "transcribe_whisper", lambda audio, proc, model, lang="ta": "hello world"
    )
    monkeypatch.setattr(B, "DEVICE", "cpu")


class TestEvaluateModelStamping:
    def test_corpus_id_covers_the_scored_samples(self, stub_whisper):
        from data.corpus import compute_corpus_id

        samples = [_corpus_sample("aaaaaaaaaaaaaaa1"), _corpus_sample("aaaaaaaaaaaaaaa2")]
        result = B.evaluate_model("whisper_small", B.MODELS["whisper_small"], samples)
        assert result["corpus_id"] == compute_corpus_id(["aaaaaaaaaaaaaaa1",
                                                         "aaaaaaaaaaaaaaa2"])

    def test_records_the_split_it_came_from(self, stub_whisper):
        samples = [_corpus_sample("aaaaaaaaaaaaaaa1")]
        result = B.evaluate_model("whisper_small", B.MODELS["whisper_small"], samples)
        assert result["split"] == "test"
        assert result["split_corpus_id"] == "sp1itc0rpus00001"

    def test_a_cap_changes_the_corpus_id(self, stub_whisper):
        samples = [_corpus_sample("aaaaaaaaaaaaaaa1"), _corpus_sample("aaaaaaaaaaaaaaa2")]
        full = B.evaluate_model("whisper_small", B.MODELS["whisper_small"], samples)
        capped = B.evaluate_model(
            "whisper_small", B.MODELS["whisper_small"], samples, max_samples=1
        )
        assert full["corpus_id"] != capped["corpus_id"]
        # Both still name the same underlying split.
        assert full["split_corpus_id"] == capped["split_corpus_id"]

    def test_unstamped_samples_yield_no_corpus_id(self, stub_whisper):
        import numpy as np

        samples = [{
            "audio": np.zeros(1600, dtype=np.float32),
            "transcript": "hello world",
            "segment_type": "monolingual_english",
        }]
        result = B.evaluate_model("whisper_small", B.MODELS["whisper_small"], samples)
        assert result["corpus_id"] is None
        assert result["split"] is None


# ---------------------------------------------------------------------------
# Merge semantics
#
# Running one model must not discard the others' rows, and a row carried over
# from an earlier run must still be checked for comparability -- merging across
# runs is how the retracted comparison was actually formed.
# ---------------------------------------------------------------------------

class TestMerge:
    def test_single_model_run_preserves_other_rows(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        combined.write_text(
            json.dumps({"whisper_tamil": _result("whisper_tamil")}), encoding="utf-8"
        )
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        rows = json.loads(combined.read_text())
        assert set(rows) == {"whisper_tamil", "whisper_small"}

    def test_rerun_replaces_that_models_row(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        stale = _result("whisper_small")
        stale["overall_wer"] = 9.9
        combined.write_text(json.dumps({"whisper_small": stale}), encoding="utf-8")
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        assert json.loads(combined.read_text())["whisper_small"]["overall_wer"] == 0.5

    def test_refuses_to_merge_into_a_different_corpus(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        previous = {"whisper_tamil": _result("whisper_tamil", "0ldc0rpus00000001")}
        combined.write_text(json.dumps(previous), encoding="utf-8")
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        assert json.loads(combined.read_text()) == previous

    def test_unreadable_combined_file_is_left_alone(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        combined.write_text("{ not json", encoding="utf-8")
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        assert combined.read_text() == "{ not json"

    def test_malformed_combined_file_is_left_alone(self, results_dir, stub_eval):
        combined = results_dir / "baseline_wer_all.json"
        combined.write_text(json.dumps({"previous": "not a result object"}),
                            encoding="utf-8")
        stub_eval()
        B.run_all_baselines([], models_to_run=["whisper_small"])
        assert json.loads(combined.read_text()) == {"previous": "not a result object"}

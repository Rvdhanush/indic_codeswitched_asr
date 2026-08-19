"""
Tests for the frozen, content-addressed corpus.

These cover the invariant the corpus exists to enforce: a test set is a fixed,
identifiable set of samples, and any drift between what two models were scored
on is detectable rather than silent.

No network. Corpora are built from small synthetic arrays under tmp_path.
"""

import json

import numpy as np
import pytest
import soundfile as sf

from data import corpus as C


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tone(seed: int, seconds: float = 0.25) -> np.ndarray:
    """Deterministic pseudo-audio in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(seconds * C.TARGET_SR)
    return rng.uniform(-0.9, 0.9, size=n).astype(np.float32)


def _sample(seed: int, transcript: str, segment_type: str = "monolingual_tamil") -> dict:
    return {
        "audio": _tone(seed),
        "transcript": transcript,
        "segment_type": segment_type,
        "switch_count": 1 if segment_type == "code_switched" else 0,
        "lang_mix_en": 0.5,
        "lang_mix_ta": 0.5,
        "duration_seconds": 0.25,
        "sample_rate": C.TARGET_SR,
        "source": {"dataset": "synthetic", "revision": "abc123", "stream_index": seed},
    }


@pytest.fixture
def splits():
    return {
        "train": [
            _sample(1, "train one"),
            _sample(2, "train two", "code_switched"),
            _sample(3, "train three", "monolingual_english"),
        ],
        "validation": [_sample(4, "val one")],
        "test": [
            _sample(5, "test one"),
            _sample(6, "test two", "code_switched"),
        ],
    }


@pytest.fixture
def frozen(tmp_path, splits):
    manifest = C.freeze(splits, corpus_dir=tmp_path)
    return tmp_path, manifest


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------

class TestSampleUid:
    def test_is_stable_across_calls(self):
        audio = _tone(7)
        assert C.sample_uid(audio, "hello") == C.sample_uid(audio, "hello")

    def test_changes_with_transcript(self):
        audio = _tone(7)
        assert C.sample_uid(audio, "hello") != C.sample_uid(audio, "hellp")

    def test_changes_with_one_audio_sample(self):
        a = _tone(7)
        b = a.copy()
        # Shift by more than one quantisation step so the change survives PCM16.
        b[100] = np.float32(b[100] + 0.01)
        assert C.sample_uid(a, "x") != C.sample_uid(b, "x")

    def test_independent_of_array_dtype_path(self):
        """A float64 view of the same signal must not change identity."""
        a = _tone(7)
        assert C.sample_uid(a, "x") == C.sample_uid(a.astype(np.float64), "x")

    def test_survives_pcm16_round_trip(self, tmp_path):
        audio = _tone(9)
        uid = C.sample_uid(audio, "round trip")
        path = tmp_path / "a.wav"
        sf.write(path, C.to_pcm16(audio), C.TARGET_SR, subtype="PCM_16", format="WAV")
        pcm, _ = sf.read(path, dtype="int16")
        assert C.sample_uid(C.from_pcm16(pcm), "round trip") == uid


class TestCorpusId:
    def test_order_independent(self):
        uids = ["aaa", "bbb", "ccc"]
        assert C.compute_corpus_id(uids) == C.compute_corpus_id(list(reversed(uids)))

    def test_membership_sensitive(self):
        assert C.compute_corpus_id(["a", "b"]) != C.compute_corpus_id(["a", "b", "c"])

    def test_empty_is_defined(self):
        assert isinstance(C.compute_corpus_id([]), str)


class TestPcm16RoundTrip:
    def test_inverse_is_exact(self):
        pcm = np.array([-32767, -1, 0, 1, 32767], dtype="<i2")
        assert np.array_equal(C.to_pcm16(C.from_pcm16(pcm)), pcm)

    def test_clips_out_of_range_input(self):
        loud = np.array([-4.0, 4.0], dtype=np.float32)
        assert C.to_pcm16(loud).tolist() == [-32767, 32767]


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

class TestFreeze:
    def test_writes_manifest_and_splits(self, frozen):
        corpus_dir, _ = frozen
        assert C.manifest_path(corpus_dir).exists()
        for name in C.SPLIT_NAMES:
            assert C.split_path(name, corpus_dir).exists()

    def test_writes_one_wav_per_sample(self, frozen):
        corpus_dir, manifest = frozen
        wavs = list(C.audio_dir(corpus_dir).glob("*.wav"))
        assert len(wavs) == manifest["n_samples"] == 6

    def test_split_corpus_ids_differ(self, frozen):
        _, manifest = frozen
        ids = manifest["split_corpus_ids"]
        assert len({ids["train"], ids["validation"], ids["test"]}) == 3

    def test_records_provenance_verbatim(self, tmp_path, splits):
        prov = {"sources": {"tamil": {"dataset": "x", "revision": "deadbeef"}}}
        manifest = C.freeze(splits, corpus_dir=tmp_path, provenance=prov)
        assert manifest["provenance"] == prov

    def test_refuses_to_overwrite_without_force(self, tmp_path, splits):
        C.freeze(splits, corpus_dir=tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            C.freeze(splits, corpus_dir=tmp_path)

    def test_force_replaces(self, tmp_path, splits):
        first = C.freeze(splits, corpus_dir=tmp_path)
        second = C.freeze(splits, corpus_dir=tmp_path, force=True)
        assert first["corpus_id"] == second["corpus_id"]

    def test_rejects_unknown_split_name(self, tmp_path, splits):
        splits["holdout"] = splits.pop("test")
        with pytest.raises(ValueError, match="Unknown split name"):
            C.freeze(splits, corpus_dir=tmp_path)

    def test_deduplicates_identical_samples_across_splits(self, tmp_path, splits):
        """
        The same clip in train and test would leak training data into the score.
        """
        splits["test"].append(splits["train"][0])
        manifest = C.freeze(splits, corpus_dir=tmp_path)
        assert manifest["n_samples"] == 6
        uids = [r["uid"] for r in manifest["samples"]]
        assert len(uids) == len(set(uids))

    def test_is_deterministic_across_builds(self, tmp_path, splits):
        a = C.freeze(splits, corpus_dir=tmp_path / "a")
        b = C.freeze(splits, corpus_dir=tmp_path / "b")
        assert a["corpus_id"] == b["corpus_id"]
        assert a["split_corpus_ids"] == b["split_corpus_ids"]

    def test_sample_order_does_not_change_corpus_id(self, tmp_path, splits):
        a = C.freeze(splits, corpus_dir=tmp_path / "a")
        shuffled = {k: list(reversed(v)) for k, v in splits.items()}
        b = C.freeze(shuffled, corpus_dir=tmp_path / "b")
        assert a["split_corpus_ids"] == b["split_corpus_ids"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

class TestLoadSplit:
    def test_round_trips_audio_bit_exactly(self, frozen, splits):
        corpus_dir, _ = frozen
        loaded = C.load_split("test", corpus_dir=corpus_dir)
        for original, got in zip(splits["test"], loaded):
            assert np.array_equal(C.to_pcm16(original["audio"]), C.to_pcm16(got["audio"]))

    def test_preserves_metadata_fields(self, frozen, splits):
        corpus_dir, _ = frozen
        loaded = C.load_split("test", corpus_dir=corpus_dir)
        for original, got in zip(splits["test"], loaded):
            for field in ("transcript", "segment_type", "switch_count",
                          "lang_mix_en", "lang_mix_ta", "sample_rate"):
                assert got[field] == original[field]

    def test_attaches_uid_and_corpus_id(self, frozen):
        corpus_dir, manifest = frozen
        loaded = C.load_split("test", corpus_dir=corpus_dir)
        assert all(s["uid"] for s in loaded)
        assert {s["corpus_id"] for s in loaded} == {manifest["split_corpus_ids"]["test"]}

    def test_uid_reproduces_from_loaded_sample(self, frozen):
        corpus_dir, _ = frozen
        for s in C.load_split("test", corpus_dir=corpus_dir):
            assert C.sample_uid(s["audio"], s["transcript"]) == s["uid"]

    def test_order_is_stable(self, frozen):
        corpus_dir, _ = frozen
        first = [s["uid"] for s in C.load_split("train", corpus_dir=corpus_dir)]
        second = [s["uid"] for s in C.load_split("train", corpus_dir=corpus_dir)]
        assert first == second

    def test_splits_are_disjoint(self, frozen):
        corpus_dir, _ = frozen
        seen = set()
        for name in C.SPLIT_NAMES:
            uids = {s["uid"] for s in C.load_split(name, corpus_dir=corpus_dir)}
            assert not (uids & seen)
            seen |= uids

    def test_splits_cover_the_manifest(self, frozen):
        corpus_dir, manifest = frozen
        covered = set()
        for name in C.SPLIT_NAMES:
            covered |= {s["uid"] for s in C.load_split(name, corpus_dir=corpus_dir)}
        assert covered == {r["uid"] for r in manifest["samples"]}

    def test_missing_corpus_names_the_freeze_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="python -m data.corpus freeze"):
            C.load_split("test", corpus_dir=tmp_path / "nothing")

    def test_rejects_unknown_split(self, frozen):
        corpus_dir, _ = frozen
        with pytest.raises(ValueError, match="Unknown split"):
            C.load_split("holdout", corpus_dir=corpus_dir)

    def test_missing_audio_is_an_error_not_a_silent_skip(self, frozen):
        corpus_dir, manifest = frozen
        uid = [r["uid"] for r in manifest["samples"] if r["split"] == "test"][0]
        C.audio_path(uid, corpus_dir).unlink()
        with pytest.raises(FileNotFoundError, match="Missing audio"):
            C.load_split("test", corpus_dir=corpus_dir)


class TestSplitCorpusId:
    def test_matches_manifest(self, frozen):
        corpus_dir, manifest = frozen
        for name in C.SPLIT_NAMES:
            assert (C.split_corpus_id(name, corpus_dir=corpus_dir)
                    == manifest["split_corpus_ids"][name])

    def test_missing_split_names_the_freeze_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="python -m data.corpus freeze"):
            C.split_corpus_id("test", corpus_dir=tmp_path / "nothing")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_clean_corpus_has_no_problems(self, frozen):
        corpus_dir, _ = frozen
        assert C.verify(corpus_dir) == []

    def test_detects_missing_corpus(self, tmp_path):
        problems = C.verify(tmp_path / "nothing")
        assert len(problems) == 1
        assert "No frozen corpus manifest" in problems[0]

    def test_detects_corrupted_audio(self, frozen, splits):
        corpus_dir, manifest = frozen
        uid = manifest["samples"][0]["uid"]
        sf.write(
            C.audio_path(uid, corpus_dir),
            C.to_pcm16(_tone(999)),
            C.TARGET_SR, subtype="PCM_16", format="WAV",
        )
        problems = C.verify(corpus_dir)
        assert any("sha256 mismatch" in p for p in problems)

    def test_detects_deleted_audio(self, frozen, splits):
        corpus_dir, manifest = frozen
        uid = manifest["samples"][0]["uid"]
        C.audio_path(uid, corpus_dir).unlink()
        assert any("audio file missing" in p for p in C.verify(corpus_dir))

    def test_detects_edited_transcript(self, frozen):
        corpus_dir, manifest = frozen
        manifest["samples"][0]["transcript"] = "silently changed"
        C.atomic_write_json(C.manifest_path(corpus_dir), manifest)
        problems = C.verify(corpus_dir)
        assert any("does not reproduce" in p for p in problems)

    def test_detects_tampered_split_corpus_id(self, frozen):
        corpus_dir, _ = frozen
        path = C.split_path("test", corpus_dir)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["corpus_id"] = "0" * 16
        C.atomic_write_json(path, doc)
        assert any("stored corpus_id" in p for p in C.verify(corpus_dir))

    def test_detects_overlapping_splits(self, frozen):
        corpus_dir, manifest = frozen
        train_uid = [r["uid"] for r in manifest["samples"] if r["split"] == "train"][0]
        path = C.split_path("test", corpus_dir)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["uids"].append(train_uid)
        doc["corpus_id"] = C.compute_corpus_id(doc["uids"])
        C.atomic_write_json(path, doc)
        assert any("must be disjoint" in p for p in C.verify(corpus_dir))

    def test_detects_orphaned_manifest_samples(self, frozen):
        corpus_dir, _ = frozen
        path = C.split_path("test", corpus_dir)
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["uids"] = doc["uids"][:1]
        doc["corpus_id"] = C.compute_corpus_id(doc["uids"])
        C.atomic_write_json(path, doc)
        assert any("belong to no split" in p for p in C.verify(corpus_dir))

    def test_no_audio_mode_skips_hashing(self, frozen):
        corpus_dir, manifest = frozen
        C.audio_path(manifest["samples"][0]["uid"], corpus_dir).unlink()
        assert C.verify(corpus_dir, check_audio=False) == []


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

class TestAtomicWriteJson:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "nested" / "x.json"
        C.atomic_write_json(path, {"a": 1})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / "x.json"
        C.atomic_write_json(path, {"a": 1})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_preserves_non_ascii(self, tmp_path):
        path = tmp_path / "x.json"
        C.atomic_write_json(path, {"t": "தமிழ்"})
        assert json.loads(path.read_text(encoding="utf-8"))["t"] == "தமிழ்"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_verify_exits_zero_on_clean_corpus(self, frozen):
        corpus_dir, _ = frozen
        assert C.main(["--corpus-dir", str(corpus_dir), "verify"]) == 0

    def test_verify_exits_nonzero_on_corruption(self, frozen):
        corpus_dir, manifest = frozen
        C.audio_path(manifest["samples"][0]["uid"], corpus_dir).unlink()
        assert C.main(["--corpus-dir", str(corpus_dir), "verify"]) == 1

    def test_info_exits_zero(self, frozen, capsys):
        corpus_dir, manifest = frozen
        assert C.main(["--corpus-dir", str(corpus_dir), "info"]) == 0
        assert manifest["corpus_id"] in capsys.readouterr().out

    def test_info_on_missing_corpus_exits_nonzero(self, tmp_path):
        assert C.main(["--corpus-dir", str(tmp_path / "nothing"), "info"]) == 1


class TestForceRebuild:
    def test_removes_orphaned_audio(self, tmp_path, splits):
        C.freeze(splits, corpus_dir=tmp_path)
        smaller = {"train": splits["train"][:1],
                   "validation": splits["validation"],
                   "test": splits["test"][:1]}
        manifest = C.freeze(smaller, corpus_dir=tmp_path, force=True)
        on_disk = {p.stem for p in C.audio_dir(tmp_path).glob("*.wav")}
        assert on_disk == {r["uid"] for r in manifest["samples"]}

    def test_rebuilt_corpus_verifies(self, tmp_path, splits):
        C.freeze(splits, corpus_dir=tmp_path)
        smaller = {"train": splits["train"][:1],
                   "validation": splits["validation"],
                   "test": splits["test"][:1]}
        C.freeze(smaller, corpus_dir=tmp_path, force=True)
        assert C.verify(tmp_path) == []

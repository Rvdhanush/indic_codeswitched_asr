"""
Frozen, content-addressed corpus.

The problem this solves
-----------------------
`data.prepare_dataset` builds a dataset by streaming from HuggingFace and
splitting it with `random_state=42`. That looks reproducible but is not: the
seed fixes the partition of whatever list it is handed, so a 300-sample pool
and a 1500-sample pool yield different test sets. The published baselines were
scored on one, the fine-tuned model on the other, and nothing recorded the
difference, so the two numbers were written into the same table.

Identity model
--------------
A sample's `uid` is derived from its content: sha256 over the exact PCM16 bytes
written to disk plus the transcript. The same clip carries the same uid no
matter which build produced it, so overlap between two corpora is measurable
rather than assumed.

A split's `corpus_id` is sha256 over its sorted uids. It is the single value
that answers "were these two models scored on the same data?", and it is
stamped into every results file.

Layout
------
    data/corpus/
      manifest.json        git-tracked  provenance + one record per sample
      splits/{train,validation,test}.json
                           git-tracked  {"corpus_id": ..., "uids": [...]}
      audio/<uid>.wav      gitignored   16 kHz mono PCM16

Audio is too large for git; the manifest carries a per-file sha256 so a rebuild
elsewhere is *verified* rather than trusted (`python -m data.corpus verify`).

CLI
---
    python -m data.corpus freeze --size 1500
    python -m data.corpus verify
    python -m data.corpus info
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

CORPUS_DIR      = Path("data/corpus")
MANIFEST_NAME   = "manifest.json"
SPLITS_DIRNAME  = "splits"
AUDIO_DIRNAME   = "audio"
SPLIT_NAMES     = ("train", "validation", "test")
SCHEMA_VERSION  = 1
UID_LEN         = 16
TARGET_SR       = 16_000

# Metadata carried per sample. `audio` is deliberately absent: it lives on disk.
_METADATA_FIELDS = (
    "transcript",
    "segment_type",
    "switch_count",
    "lang_mix_en",
    "lang_mix_ta",
    "duration_seconds",
    "sample_rate",
)

_FREEZE_HINT = "Build one with: python -m data.corpus freeze"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def manifest_path(corpus_dir: Path = CORPUS_DIR) -> Path:
    return Path(corpus_dir) / MANIFEST_NAME


def splits_dir(corpus_dir: Path = CORPUS_DIR) -> Path:
    return Path(corpus_dir) / SPLITS_DIRNAME


def split_path(name: str, corpus_dir: Path = CORPUS_DIR) -> Path:
    return splits_dir(corpus_dir) / f"{name}.json"


def audio_dir(corpus_dir: Path = CORPUS_DIR) -> Path:
    return Path(corpus_dir) / AUDIO_DIRNAME


def audio_path(uid: str, corpus_dir: Path = CORPUS_DIR) -> Path:
    return audio_dir(corpus_dir) / f"{uid}.wav"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, payload) -> None:
    """
    Write JSON via a temp file + os.replace so an interrupted or failing run can
    never leave a truncated (or empty) file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Content addressing
# ---------------------------------------------------------------------------

def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """
    Canonical on-disk representation: mono little-endian int16.

    Everything hashes and round-trips through this, so `uid` is stable across a
    write/read cycle. Scaling by 32767 (not 32768) keeps the inverse exact:
    int16 -> /32767 -> *32767 -> round returns the original integer.
    """
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=0)
    a = np.clip(a, -1.0, 1.0)
    return np.rint(a * 32767.0).astype("<i2")


def from_pcm16(pcm: np.ndarray) -> np.ndarray:
    """Inverse of `to_pcm16`."""
    return (np.asarray(pcm, dtype=np.float32) / 32767.0).astype(np.float32)


def sample_uid(audio: np.ndarray, transcript: str) -> str:
    """
    Content-addressed id for one sample. Depends only on the audio samples and
    the transcript, never on build order, pool size, or filesystem path.
    """
    h = hashlib.sha256()
    h.update(to_pcm16(audio).tobytes())
    h.update(b"\x00")
    h.update(transcript.encode("utf-8"))
    return h.hexdigest()[:UID_LEN]


def compute_corpus_id(uids: Iterable[str]) -> str:
    """
    Id for a set of samples. Order-independent, so two builds that produced the
    same samples in a different order still compare equal.
    """
    joined = "\n".join(sorted(uids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:UID_LEN]


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def freeze(
    splits: dict,
    corpus_dir: Path = CORPUS_DIR,
    provenance: Optional[dict] = None,
    force: bool = False,
) -> dict:
    """
    Write `splits` to disk as a frozen corpus.

    Args:
        splits:     {"train": [sample, ...], "validation": [...], "test": [...]}
                    where each sample is a dict from `data.prepare_dataset`
                    (keys: audio, transcript, segment_type, ...).
        provenance: recorded verbatim into the manifest: dataset revisions,
                    builder constants, split ratios.
        force:      required to replace an existing manifest. Silently
                    overwriting a frozen corpus is the failure mode this module
                    exists to prevent.

    Returns the manifest that was written.
    """
    corpus_dir = Path(corpus_dir)
    mpath = manifest_path(corpus_dir)
    if mpath.exists() and not force:
        raise FileExistsError(
            f"A frozen corpus already exists at {mpath}. Results in results/ may "
            f"reference its corpus_id; replacing it silently invalidates them. "
            f"Pass --force (or force=True) if that is what you want."
        )

    unknown = [k for k in splits if k not in SPLIT_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown split name(s): {unknown}. Expected {list(SPLIT_NAMES)}."
        )

    audio_dir(corpus_dir).mkdir(parents=True, exist_ok=True)
    splits_dir(corpus_dir).mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    split_uids: dict[str, list[str]] = {}
    duplicates = 0

    for split_name in SPLIT_NAMES:
        samples = splits.get(split_name, [])
        uids: list[str] = []

        for sample in samples:
            audio = np.asarray(sample["audio"], dtype=np.float32)
            transcript = sample["transcript"]
            uid = sample_uid(audio, transcript)

            if uid in records:
                # Identical content already frozen. Keeping both copies would
                # put the same clip in two splits and leak train into test.
                duplicates += 1
                continue

            dest = audio_path(uid, corpus_dir)
            sf.write(dest, to_pcm16(audio), TARGET_SR, subtype="PCM_16", format="WAV")

            record = {"uid": uid, "split": split_name}
            for field in _METADATA_FIELDS:
                if field in sample:
                    record[field] = sample[field]
            record["sample_rate"] = TARGET_SR
            if "source" in sample:
                record["source"] = sample["source"]
            record["audio_sha256"] = _file_sha256(dest)
            record["n_audio_samples"] = int(len(audio))

            records[uid] = record
            uids.append(uid)

        split_uids[split_name] = uids

    if duplicates:
        logger.warning(
            "Dropped %d duplicate sample(s): identical audio+transcript already "
            "present. Splits below reflect the deduplicated set.", duplicates
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": len(records),
        "corpus_id": compute_corpus_id(records.keys()),
        "split_corpus_ids": {
            name: compute_corpus_id(uids) for name, uids in split_uids.items()
        },
        "provenance": provenance or {},
        "samples": [records[uid] for uid in records],
    }
    atomic_write_json(mpath, manifest)

    for name, uids in split_uids.items():
        atomic_write_json(split_path(name, corpus_dir), {
            "split": name,
            "corpus_id": compute_corpus_id(uids),
            "n_samples": len(uids),
            "uids": uids,
        })

    stale = [
        path for path in audio_dir(corpus_dir).glob("*.wav")
        if path.stem not in records
    ]
    for path in stale:
        path.unlink()
    if stale:
        logger.info("Removed %d orphaned audio file(s) from a previous corpus.",
                    len(stale))

    logger.info(
        "Froze %d samples to %s (corpus_id %s)",
        len(records), corpus_dir, manifest["corpus_id"],
    )
    return manifest


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_manifest(corpus_dir: Path = CORPUS_DIR) -> dict:
    mpath = manifest_path(corpus_dir)
    if not mpath.exists():
        raise FileNotFoundError(
            f"No frozen corpus manifest at {mpath}. {_FREEZE_HINT}"
        )
    return _read_json(mpath)


def split_corpus_id(name: str, corpus_dir: Path = CORPUS_DIR) -> str:
    """Read a split's corpus_id without loading any audio."""
    spath = split_path(name, corpus_dir)
    if not spath.exists():
        raise FileNotFoundError(
            f"No frozen '{name}' split at {spath}. {_FREEZE_HINT}"
        )
    return _read_json(spath)["corpus_id"]


def load_split(name: str, corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """
    Load one frozen split as sample dicts in the shape the rest of the pipeline
    already expects: `audio` (float32 @ 16 kHz), `transcript`, `segment_type`,
    `switch_count`, plus `uid` and `corpus_id`.

    Returned in the order recorded in the split file, so two runs iterate the
    same samples in the same order.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(
            f"Unknown split '{name}'. Expected one of {list(SPLIT_NAMES)}."
        )

    spath = split_path(name, corpus_dir)
    if not spath.exists():
        raise FileNotFoundError(
            f"No frozen '{name}' split at {spath}. {_FREEZE_HINT}"
        )

    split_doc = _read_json(spath)
    manifest = load_manifest(corpus_dir)
    by_uid = {r["uid"]: r for r in manifest["samples"]}

    samples = []
    for uid in split_doc["uids"]:
        record = by_uid.get(uid)
        if record is None:
            raise KeyError(
                f"uid {uid} is in {spath} but not in the manifest. The corpus is "
                f"inconsistent; run: python -m data.corpus verify"
            )

        apath = audio_path(uid, corpus_dir)
        if not apath.exists():
            raise FileNotFoundError(
                f"Missing audio for uid {uid} at {apath}. Audio is not tracked in "
                f"git; rebuild it with: python -m data.corpus freeze --force"
            )

        # Read as int16 explicitly. soundfile's float conversion divides by
        # 32768, which would break the uid round-trip.
        pcm, sr = sf.read(apath, dtype="int16", always_2d=False)
        if sr != TARGET_SR:
            raise ValueError(f"{apath} has sample rate {sr}, expected {TARGET_SR}")

        sample = {
            k: v for k, v in record.items()
            if k not in ("audio_sha256", "n_audio_samples")
        }
        sample["audio"] = from_pcm16(pcm)
        sample["corpus_id"] = split_doc["corpus_id"]
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(corpus_dir: Path = CORPUS_DIR, check_audio: bool = True) -> list[str]:
    """
    Check a frozen corpus for internal consistency. Returns a list of problems;
    an empty list means the corpus is intact.

    Checks: every manifest sample has an audio file whose sha256 matches; every
    uid is reproducible from its audio + transcript; splits are disjoint and
    cover the manifest exactly; each stored corpus_id matches a recomputation.
    """
    corpus_dir = Path(corpus_dir)
    problems: list[str] = []

    try:
        manifest = load_manifest(corpus_dir)
    except FileNotFoundError as e:
        return [str(e)]

    by_uid = {r["uid"]: r for r in manifest["samples"]}
    if len(by_uid) != len(manifest["samples"]):
        problems.append("manifest contains duplicate uids")

    if check_audio:
        for uid, record in by_uid.items():
            apath = audio_path(uid, corpus_dir)
            if not apath.exists():
                problems.append(f"{uid}: audio file missing at {apath}")
                continue
            actual = _file_sha256(apath)
            if actual != record.get("audio_sha256"):
                problems.append(
                    f"{uid}: audio sha256 mismatch "
                    f"(manifest {record.get('audio_sha256')}, on disk {actual})"
                )
                continue
            pcm, _ = sf.read(apath, dtype="int16", always_2d=False)
            recomputed = sample_uid(from_pcm16(pcm), record["transcript"])
            if recomputed != uid:
                problems.append(
                    f"{uid}: content hash does not reproduce (got {recomputed}); "
                    f"audio or transcript was edited after freezing"
                )

    seen: dict[str, str] = {}
    covered: set[str] = set()
    for name in SPLIT_NAMES:
        spath = split_path(name, corpus_dir)
        if not spath.exists():
            problems.append(f"missing split file {spath}")
            continue
        doc = _read_json(spath)
        uids = doc["uids"]

        if len(set(uids)) != len(uids):
            problems.append(f"split '{name}' contains duplicate uids")

        recomputed = compute_corpus_id(uids)
        if doc.get("corpus_id") != recomputed:
            problems.append(
                f"split '{name}': stored corpus_id {doc.get('corpus_id')} != "
                f"recomputed {recomputed}"
            )

        for uid in uids:
            if uid not in by_uid:
                problems.append(f"split '{name}': uid {uid} not in manifest")
            if uid in seen:
                problems.append(
                    f"uid {uid} appears in both '{seen[uid]}' and '{name}'; "
                    f"splits must be disjoint"
                )
            else:
                seen[uid] = name
            covered.add(uid)

    orphans = set(by_uid) - covered
    if orphans:
        problems.append(
            f"{len(orphans)} manifest sample(s) belong to no split "
            f"(e.g. {sorted(orphans)[:3]})"
        )

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_freeze(args) -> int:
    from data.prepare_dataset import (
        authenticate_hf,
        build_dataset_splits,
        build_provenance,
        load_indicvoices_tamil,
        resolve_dataset_revision,
        TAMIL_DATASET,
        ENGLISH_DATASET,
    )

    authenticate_hf()

    tamil_rev = args.tamil_revision or resolve_dataset_revision(TAMIL_DATASET)
    english_rev = args.english_revision or resolve_dataset_revision(ENGLISH_DATASET)

    samples = load_indicvoices_tamil(
        max_samples=args.size,
        tamil_revision=tamil_rev,
        english_revision=english_rev,
    )
    splits = build_dataset_splits(samples)

    provenance = build_provenance(
        tamil_revision=tamil_rev,
        english_revision=english_rev,
        size=args.size,
    )
    try:
        manifest = freeze(
            splits,
            corpus_dir=args.corpus_dir,
            provenance=provenance,
            force=args.force,
        )
    except FileExistsError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nFrozen {manifest['n_samples']} samples to {args.corpus_dir}")
    for name, cid in manifest["split_corpus_ids"].items():
        n = sum(1 for r in manifest["samples"] if r["split"] == name)
        print(f"  {name:<12} {n:>5} samples  corpus_id {cid}")
    if not tamil_rev or not english_rev:
        print(
            "\nWARNING: at least one dataset revision could not be resolved. "
            "A rebuild is not guaranteed to reproduce these corpus_ids."
        )
    return 0


def _cmd_verify(args) -> int:
    problems = verify(args.corpus_dir, check_audio=not args.no_audio)
    if problems:
        print(f"FAILED: {len(problems)} problem(s) in {args.corpus_dir}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK: corpus at {args.corpus_dir} is internally consistent.")
    return 0


def _cmd_info(args) -> int:
    from collections import Counter

    try:
        manifest = load_manifest(args.corpus_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    prov = manifest.get("provenance", {})
    print(f"Corpus:      {args.corpus_dir}")
    print(f"corpus_id:   {manifest['corpus_id']}")
    print(f"created:     {manifest.get('created_utc', '-')}")
    print(f"samples:     {manifest['n_samples']}")

    sources = prov.get("sources", {})
    if sources:
        print("\nSources:")
        for key, src in sources.items():
            rev = src.get("revision") or "(unpinned)"
            print(f"  {key:<8} {src.get('dataset')}  revision {rev}")

    print("\nSplits:")
    for name in SPLIT_NAMES:
        rows = [r for r in manifest["samples"] if r["split"] == name]
        cid = manifest["split_corpus_ids"].get(name, "-")
        types = Counter(r.get("segment_type", "unknown") for r in rows)
        breakdown = ", ".join(f"{t}={c}" for t, c in sorted(types.items()))
        print(f"  {name:<12} {len(rows):>5} samples  corpus_id {cid}")
        if breakdown:
            print(f"               {breakdown}")
    return 0


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m data.corpus",
        description="Build, verify and inspect the frozen evaluation corpus.",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=CORPUS_DIR,
        help=f"Corpus root (default: {CORPUS_DIR}).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_freeze = sub.add_parser("freeze", help="Build from HuggingFace and freeze to disk.")
    p_freeze.add_argument(
        "--size", type=int, default=1500,
        help="Total samples to build before splitting (default: 1500).",
    )
    p_freeze.add_argument(
        "--force", action="store_true",
        help="Replace an existing frozen corpus.",
    )
    p_freeze.add_argument(
        "--tamil-revision", default=None,
        help="Pin the Tamil dataset to this revision (default: resolve current).",
    )
    p_freeze.add_argument(
        "--english-revision", default=None,
        help="Pin the English dataset to this revision (default: resolve current).",
    )
    p_freeze.set_defaults(func=_cmd_freeze)

    p_verify = sub.add_parser("verify", help="Check integrity of a frozen corpus.")
    p_verify.add_argument(
        "--no-audio", action="store_true",
        help="Skip per-file hashing (fast structural check only).",
    )
    p_verify.set_defaults(func=_cmd_verify)

    p_info = sub.add_parser("info", help="Print corpus ids, provenance and counts.")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

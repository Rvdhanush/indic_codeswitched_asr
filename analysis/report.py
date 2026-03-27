"""
Failure analysis report generator.

Reads results/baseline_wer_all.json and produces:
  - results/failure_analysis_report.md   (human-readable)
  - results/failure_analysis_summary.json (machine-readable)

Usage:
    python analysis/report.py
    python analysis/report.py --results results/baseline_wer_all.json
"""

import json
import argparse
from pathlib import Path

RESULTS_DIR = Path("results")

FAILURE_LABELS = {
    "SUBSTITUTION_SWITCH": "Error at language switch boundary",
    "DELETION_PROPER_NOUN": "Proper noun deleted",
    "SUBSTITUTION_NUMBER": "Number / date transcribed incorrectly",
    "LANGUAGE_CONFUSION": "Wrong language script used",
    "INSERTION_FILLER": "Hallucinated filler word",
}

MODEL_DISPLAY = {
    "whisper_medium": "Whisper-medium",
    "indic_whisper": "IndicWhisper",
    "indicwav2vec": "IndicWav2Vec",
}


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def code_switch_penalty(r: dict) -> float | None:
    """
    Ratio of code-switched WER to average monolingual WER.
    Values > 1 mean the model degrades on code-switched speech.
    Returns None if either WER field is missing.
    """
    cs = r.get("code_switched_wer")
    mono_ta = r.get("monolingual_tamil_wer")
    mono_en = r.get("monolingual_english_wer")

    available = [v for v in (mono_ta, mono_en) if v is not None]
    if not available or cs is None:
        return None

    avg_mono = sum(available) / len(available)
    if avg_mono == 0:
        return None
    return round(cs / avg_mono, 3)


def dominant_failure(breakdown: dict) -> tuple[str, float]:
    """Return (category, share_pct) for the most frequent failure type."""
    total = sum(breakdown.values())
    if total == 0:
        return ("N/A", 0.0)
    category = max(breakdown, key=breakdown.get)
    share = round(breakdown[category] / total * 100, 1)
    return (category, share)


def shared_failures(all_results: dict) -> list[str]:
    """
    Return failure categories where all models rank it in their top-2.
    These represent systemic weaknesses across architectures.
    """
    top2_per_model = []
    for r in all_results.values():
        bd = r.get("failure_breakdown", {})
        if not bd:
            continue
        sorted_cats = sorted(bd, key=bd.get, reverse=True)
        top2_per_model.append(set(sorted_cats[:2]))

    if not top2_per_model:
        return []
    shared = top2_per_model[0]
    for s in top2_per_model[1:]:
        shared = shared & s
    return sorted(shared)


def wer_ranking(all_results: dict, field: str) -> list[tuple[str, float]]:
    """Return models sorted by a WER field (ascending)."""
    rows = [
        (k, v[field])
        for k, v in all_results.items()
        if v.get(field) is not None
    ]
    return sorted(rows, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return "| " + " | ".join(
            str(c).ljust(col_widths[i]) for i, c in enumerate(cells)
        ) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def build_markdown(all_results: dict) -> str:
    lines = []

    lines += [
        "# Tamil-English Code-Switched ASR: Failure Analysis Report",
        "",
        "Generated from `results/baseline_wer_all.json`.",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. WER comparison table
    # ------------------------------------------------------------------
    lines += ["## 1. WER by Segment Type", ""]

    headers = ["Model", "Overall WER", "Mono-Tamil WER", "Mono-English WER",
               "Code-Switched WER", "CS Penalty"]
    rows = []
    for key, r in all_results.items():
        label = MODEL_DISPLAY.get(key, key)
        penalty = code_switch_penalty(r)
        penalty_str = f"{penalty:.2f}×" if penalty is not None else "—"
        rows.append([
            label,
            _fmt(r.get("overall_wer")),
            _fmt(r.get("monolingual_tamil_wer")),
            _fmt(r.get("monolingual_english_wer")),
            _fmt(r.get("code_switched_wer")),
            penalty_str,
        ])

    lines += [_md_table(headers, rows), ""]
    lines += [
        "> **CS Penalty** = code-switched WER ÷ average monolingual WER. "
        "A value of 2.0× means the model makes twice as many errors on "
        "code-switched speech as on clean monolingual speech.",
        "",
    ]

    # ------------------------------------------------------------------
    # 2. Code-switched ranking
    # ------------------------------------------------------------------
    lines += ["## 2. Model Ranking on Code-Switched Speech", ""]
    cs_rank = wer_ranking(all_results, "code_switched_wer")
    if cs_rank:
        rank_rows = [
            [i + 1, MODEL_DISPLAY.get(k, k), _fmt(wer)]
            for i, (k, wer) in enumerate(cs_rank)
        ]
        lines += [_md_table(["Rank", "Model", "Code-Switched WER"], rank_rows), ""]

        overall_rank = wer_ranking(all_results, "overall_wer")
        overall_order = [k for k, _ in overall_rank]
        cs_order = [k for k, _ in cs_rank]
        if overall_order != cs_order:
            lines += [
                "> **Note:** Model ranking on code-switched speech differs from "
                "overall ranking — overall WER alone is misleading for this task.",
                "",
            ]

    # ------------------------------------------------------------------
    # 3. Failure breakdown per model
    # ------------------------------------------------------------------
    lines += ["## 3. Failure Category Breakdown", ""]

    bd_headers = ["Failure Category", "Description"] + [
        MODEL_DISPLAY.get(k, k) for k in all_results
    ]
    all_cats = list(FAILURE_LABELS.keys())
    totals = {
        k: sum(r.get("failure_breakdown", {}).values())
        for k, r in all_results.items()
    }

    bd_rows = []
    for cat in all_cats:
        row = [f"`{cat}`", FAILURE_LABELS[cat]]
        for key, r in all_results.items():
            count = r.get("failure_breakdown", {}).get(cat, 0)
            total = totals[key]
            pct = f"{count / total * 100:.0f}%" if total else "—"
            row.append(f"{count} ({pct})")
        bd_rows.append(row)

    lines += [_md_table(bd_headers, bd_rows), ""]

    # Dominant failure per model
    lines += ["**Dominant failure per model:**", ""]
    for key, r in all_results.items():
        bd = r.get("failure_breakdown", {})
        cat, share = dominant_failure(bd)
        label = MODEL_DISPLAY.get(key, key)
        lines.append(
            f"- **{label}:** `{cat}` — {FAILURE_LABELS.get(cat, cat)} "
            f"({share}% of all failures)"
        )
    lines.append("")

    # Shared failures
    shared = shared_failures(all_results)
    if shared:
        lines += [
            "**Systemic failures (top-2 for all models):**",
            "",
        ]
        for cat in shared:
            lines.append(f"- `{cat}` — {FAILURE_LABELS.get(cat, cat)}")
        lines += [
            "",
            "> These categories represent architectural blind spots shared across "
            "Whisper, IndicWhisper, and IndicWav2Vec — not model-specific bugs. "
            "They are the highest-leverage targets for fine-tuning data curation.",
            "",
        ]

    # ------------------------------------------------------------------
    # 4. Fine-tuning implications
    # ------------------------------------------------------------------
    lines += [
        "## 4. Fine-tuning Implications",
        "",
        "The failure breakdown directly informs the data sampling strategy in "
        "`fine_tuning/train.py`:",
        "",
    ]

    implication_rows = [
        ["`SUBSTITUTION_SWITCH`",
         "Oversample segments with high switch-point count (×2 in config)"],
        ["`DELETION_PROPER_NOUN`",
         "Include samples with named entities; avoid aggressive text normalisation"],
        ["`SUBSTITUTION_NUMBER`",
         "Ensure numeric utterances are present in training mix"],
        ["`LANGUAGE_CONFUSION`",
         "Oversample code-switched segments overall (×3 in config)"],
        ["`INSERTION_FILLER`",
         "Use `RemovePunctuation` + `ToLowerCase` transforms to reduce hallucination surface"],
    ]
    lines += [
        _md_table(
            ["Failure Category", "Mitigation in fine_tuning/train.py"],
            implication_rows
        ),
        "",
    ]

    # ------------------------------------------------------------------
    # 5. Sample counts
    # ------------------------------------------------------------------
    lines += ["## 5. Evaluation Coverage", ""]
    cov_rows = [
        [
            MODEL_DISPLAY.get(k, k),
            r.get("total_samples", "—"),
            r.get("errors", "—"),
            r.get("device", "—"),
        ]
        for k, r in all_results.items()
    ]
    lines += [
        _md_table(["Model", "Samples evaluated", "Errors", "Device"], cov_rows),
        "",
    ]

    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def build_summary(all_results: dict) -> dict:
    summary = {}
    for key, r in all_results.items():
        bd = r.get("failure_breakdown", {})
        dom_cat, dom_share = dominant_failure(bd)
        summary[key] = {
            "model_name": r.get("model_name"),
            "overall_wer": r.get("overall_wer"),
            "monolingual_tamil_wer": r.get("monolingual_tamil_wer"),
            "monolingual_english_wer": r.get("monolingual_english_wer"),
            "code_switched_wer": r.get("code_switched_wer"),
            "code_switch_penalty": code_switch_penalty(r),
            "dominant_failure_category": dom_cat,
            "dominant_failure_share_pct": dom_share,
            "failure_breakdown": bd,
        }
    summary["_meta"] = {
        "shared_systemic_failures": shared_failures(all_results),
        "cs_wer_ranking": [k for k, _ in wer_ranking(all_results, "code_switched_wer")],
        "overall_wer_ranking": [k for k, _ in wer_ranking(all_results, "overall_wer")],
    }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate failure analysis report")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR / "baseline_wer_all.json",
        help="Path to baseline_wer_all.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to write report files",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: results file not found at {args.results}")
        print("Run evaluation/baseline_eval.py first.")
        raise SystemExit(1)

    with open(args.results) as f:
        all_results = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    md = build_markdown(all_results)
    md_path = args.out_dir / "failure_analysis_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Report written to {md_path}")

    summary = build_summary(all_results)
    json_path = args.out_dir / "failure_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {json_path}")


if __name__ == "__main__":
    main()

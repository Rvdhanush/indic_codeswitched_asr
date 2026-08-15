# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Comparative evaluation and LoRA fine-tuning project for Tamil-English code-switched Automatic Speech Recognition (ASR). Evaluates three pre-trained models (Whisper, IndicWhisper, IndicWav2Vec) and fine-tunes Whisper-small with targeted oversampling of code-switched data.

## Setup

The project is an installable package (`pyproject.toml`). Install it editable — first-party
imports do not resolve otherwise.

```bash
pip install -e ".[train]"   # or ".[serve]" for API only, ".[dev]" for tests
cp .env.example .env
# Set HF_TOKEN and WANDB_API_KEY in .env
```

Dependency sets live in `pyproject.toml` optional-dependencies, mirrored by
`requirements/{base,serve,train,dev}.txt`. The top-level `requirements.txt` and
`requirements-dev.txt` are thin pointers kept for CI and the Colab notebook.

## Commands

Always run as modules. `python evaluation/baseline_eval.py` fails — running a file as a script
puts `evaluation/` on `sys.path[0]`, so `from evaluation.metrics import ...` cannot resolve.

**Prepare dataset** (streams from HuggingFace, outputs metadata to `data/processed/`):
```bash
python -m data.prepare_dataset
```

**Run baseline evaluation** (outputs to `results/`):
```bash
python -m evaluation.baseline_eval --dry-run   # list models, write nothing
python -m evaluation.baseline_eval             # all three baselines
python -m evaluation.baseline_eval --models whisper_small --max-samples 10
```

**Fine-tune with LoRA** (config in `fine_tuning/config.yaml`, checkpoints to `checkpoints/best_model/`):
```bash
python -m fine_tuning.train
```

**Failure analysis report**:
```bash
python -m analysis.report
```

**Tests**:
```bash
python -m pytest tests/ -q
```

## Architecture

### Pipeline Flow
1. **Data** (`data/prepare_dataset.py`) — Builds a mixed dataset via synthetic code-switching: loads Tamil segments from IndicVoices-R and English segments from LibriSpeech, concatenates Tamil+silence+English pairs to create code-switched samples, resamples to 16kHz, caps segments at 8s, tags each as `monolingual_tamil` / `monolingual_english` / `code_switched`, stratified 80/10/10 split.

2. **Evaluation** (`evaluation/baseline_eval.py` + `evaluation/metrics.py`) — Loads all three baseline models, transcribes the test set, computes WER/CER stratified by segment type, and categorizes failures into 5 types: `SUBSTITUTION_SWITCH`, `DELETION_PROPER_NOUN`, `SUBSTITUTION_NUMBER`, `LANGUAGE_CONFUSION`, `INSERTION_FILLER`.

3. **Fine-tuning** (`fine_tuning/train.py`) — Applies LoRA adapters to Whisper-small (`q_proj`, `v_proj`; r=32, alpha=64), uses a custom weighted sampler that oversamples code-switched 3×, high-switch-point samples a further 2× (6× total), and undersamples monolingual to 50%. Trains via HuggingFace `Seq2SeqTrainer` with FP16, AdamW 8-bit, and WandB logging. `build_training_args()` is shared with the Colab notebook — do not duplicate it there.

## Known Defects (do not build on these without reading)

The published results are **not** currently reproducible or valid as a comparison. Before
changing anything in the eval or data path, read `README.md` → Known Limitations. Summary:

- **Baselines and the fine-tuned model were scored on different test sets** — the headline
  "41% reduction" is unsupported and has been retracted in all docs.
- **`_make_cs_sample` does not produce code-switching** — it concatenates a whole Tamil
  utterance and a whole English utterance from different corpora, and hardcodes
  `switch_count=1`. Channel change is confounded with the label.
- **`compute_stratified_wer` macro-averages per-utterance WER** rather than computing
  corpus-level `total_errors / total_words`.
- **The 5-category failure taxonomy has 2 live categories.** `LANGUAGE_CONFUSION` is the
  fallback return; `DELETION_PROPER_NOUN` is unreachable (reference is lowercased before an
  `isupper()` test); `SUBSTITUTION_NUMBER` never fires. `categorize_failure` also compares words
  positionally with no alignment.
- **`checkpoints/` is gitignored** and `api/app.py` loads the adapter only from a local path,
  falling back to base Whisper-small while still reporting `/health` 200. A clone-based deploy
  silently serves the un-finetuned model.

The roadmap for fixing these is a frozen content-addressed corpus, corpus-level + script-normalized
WER, a real-speech eval set, and intra-sentential synthesis. Do not publish new numbers from the
current pipeline.

### Key Design Decisions
- **Streaming dataset loading** — IndicVoices is large; `load_indicvoices_tamil()` uses `streaming=True` to avoid downloading the full corpus.
- **Segment-type stratification** — Both the train/val/test split and the training sampler are stratified by segment type to ensure code-switched samples are represented despite being a minority class.
- **LoRA on attention only** — Only `q_proj`/`v_proj` are adapted; encoder, decoder embeddings, and LM head are frozen, keeping trainable parameters small.
- **Failure taxonomy** — The 5-category failure taxonomy in `metrics.py` directly motivates the oversampling strategy in `train.py`.

### Additional Modules
- **`analysis/report.py`** — Reads `results/baseline_wer_all.json`, computes derived metrics (CS penalty, dominant failure, shared failures), and writes `results/failure_analysis_report.md` + `results/failure_analysis_summary.json`. Run with `python -m analysis.report`.
- **`api/app.py`** — FastAPI inference server with `/health`, `/transcribe`, `/analyze`, `/model/info` endpoints. Loads fine-tuned LoRA checkpoint if available, falls back to base Whisper-small. Run with `uvicorn api.app:app --host 0.0.0.0 --port 8000`.
- **`notebooks/colab_finetune.ipynb`** — Colab-ready notebook for end-to-end fine-tuning on GPU.

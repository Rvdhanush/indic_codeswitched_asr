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

**Build the frozen corpus** (streams from HuggingFace, writes `data/corpus/`). This is the
dataset entrypoint — everything downstream reads the frozen splits, nothing rebuilds:
```bash
python -m data.corpus freeze --size 1500   # refuses to overwrite without --force
python -m data.corpus verify               # re-hash every wav, check split disjointness
python -m data.corpus info                 # corpus_id per split, pinned revisions
```

`python -m data.prepare_dataset` still exists but only writes an ad-hoc metadata dump to the
gitignored `data/processed/`. Nothing reads it.

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

## Site and branches

`master` is **frozen** at `65505f5` and must stay that way unless the user says otherwise. It still
carries the retracted "41% WER reduction" headline; that is deliberate, not an oversight.

All work lands on `dev`. The public site publishes from `dev`:

```
commit to dev -> push origin dev -> .github/workflows/pages.yml -> live (~17s)
```

- **Live:** https://rvdhanush.github.io/indic_codeswitched_asr/
- **Source:** `docs/index.html` — one self-contained file, no build step. Kept single-file so the
  same source feeds both the design preview and production. Revisit a framework only when the
  hosted demo adds real interactive state (upload, loading, error, comparison).
- Pages is configured with `build_type: workflow`, so the Pages branch setting is ignored.

The site currently presents `master`'s numbers, and its footer links point at `/blob/master/...`,
which is frozen — so the site and everything it links to agree. If the site is updated to the
corrected results, those links move to `dev` **in the same change**, never before.

## Architecture

### Pipeline Flow
1. **Data** (`data/prepare_dataset.py`) — Builds a mixed dataset via synthetic code-switching: loads Tamil segments from IndicVoices-R and English segments from LibriSpeech, concatenates Tamil+silence+English pairs to create code-switched samples, resamples to 16kHz, caps segments at 8s, tags each as `monolingual_tamil` / `monolingual_english` / `code_switched`, stratified 80/10/10 split.

2. **Corpus** (`data/corpus.py`) — Freezes a build to disk under `data/corpus/`. Each sample gets
   `uid = sha256(pcm16 bytes + transcript)`; each split gets `corpus_id = sha256(sorted uids)`.
   `manifest.json` and `splits/*.json` are git-tracked; `audio/` is gitignored and verified by
   per-file sha256. `load_split(name)` returns samples in the shape the rest of the pipeline
   already expects, plus `uid` and `corpus_id`.

3. **Evaluation** (`evaluation/baseline_eval.py` + `evaluation/metrics.py`) — Reads a frozen split,
   loads all three baseline models, transcribes the test set, computes WER/CER stratified by segment type, and categorizes failures into 5 types: `SUBSTITUTION_SWITCH`, `DELETION_PROPER_NOUN`, `SUBSTITUTION_NUMBER`, `LANGUAGE_CONFUSION`, `INSERTION_FILLER`.

4. **Fine-tuning** (`fine_tuning/train.py`) — Reads the frozen `train`/`validation` splits and
   applies LoRA adapters to Whisper-small (`q_proj`, `v_proj`; r=32, alpha=64), uses a custom weighted sampler that oversamples code-switched 3×, high-switch-point samples a further 2× (6× total), and undersamples monolingual to 50%. Trains via HuggingFace `Seq2SeqTrainer` with FP16, AdamW 8-bit, and WandB logging. `build_training_args()` is shared with the Colab notebook — do not duplicate it there.

## Known Defects (do not build on these without reading)

The published results are **not** currently reproducible or valid as a comparison. Before
changing anything in the eval or data path, read `README.md` → Known Limitations. Summary:

- ~~**Baselines and the fine-tuned model were scored on different test sets**~~ — **fixed.**
  Evaluation and training now read a frozen content-addressed corpus (`data/corpus.py`,
  `data/corpus/`); results carry the `corpus_id` of the samples they were scored on, and both
  `run_all_baselines` and `analysis/report.py` refuse to compare mismatched ids. The published
  numbers still predate this and remain retracted until every model is re-run on one split.
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

The remaining roadmap is corpus-level + script-normalized WER, an alignment-driven failure
taxonomy, intra-sentential synthesis, and a real-speech eval set. Do not publish new numbers
from the current pipeline.

Note on synthesis: fixing `_make_cs_sample` changes the audio and therefore every `uid`. It must
land as a new frozen corpus, not as an edit to the existing one.

### Key Design Decisions
- **Frozen corpus over rebuild-on-run** — a seeded split is only reproducible against a fixed input
  pool, so any code path that rebuilds before scoring can silently change the test set. The corpus
  is built once and addressed by content thereafter.
- **Streaming dataset loading** — IndicVoices is large; `load_indicvoices_tamil()` uses `streaming=True` to avoid downloading the full corpus.
- **Segment-type stratification** — Both the train/val/test split and the training sampler are stratified by segment type to ensure code-switched samples are represented despite being a minority class.
- **LoRA on attention only** — Only `q_proj`/`v_proj` are adapted; encoder, decoder embeddings, and LM head are frozen, keeping trainable parameters small.
- **Failure taxonomy** — The 5-category failure taxonomy in `metrics.py` directly motivates the oversampling strategy in `train.py`.

### Additional Modules
- **`analysis/report.py`** — Reads `results/baseline_wer_all.json`, computes derived metrics (CS penalty, dominant failure, shared failures), and writes `results/failure_analysis_report.md` + `results/failure_analysis_summary.json`. Run with `python -m analysis.report`.
- **`api/app.py`** — FastAPI inference server with `/health`, `/transcribe`, `/analyze`, `/model/info` endpoints. Loads fine-tuned LoRA checkpoint if available, falls back to base Whisper-small. Run with `uvicorn api.app:app --host 0.0.0.0 --port 8000`.
- **`notebooks/colab_finetune.ipynb`** — Colab-ready notebook for end-to-end fine-tuning on GPU.

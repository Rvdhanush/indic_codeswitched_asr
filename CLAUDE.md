# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Comparative evaluation and LoRA fine-tuning project for Tamil-English code-switched Automatic Speech Recognition (ASR). Evaluates three pre-trained models (Whisper, IndicWhisper, IndicWav2Vec) and fine-tunes Whisper-small with targeted oversampling of code-switched data.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Set HF_TOKEN and WANDB_API_KEY in .env
```

## Commands

**Prepare dataset** (streams from HuggingFace IndicVoices Tamil, outputs to `data/processed/`):
```bash
python data/prepare_dataset.py
```

**Run baseline evaluation** (outputs to `results/`):
```bash
python evaluation/baseline_eval.py
```

**Fine-tune with LoRA** (config in `fine_tuning/config.yaml`, checkpoints to `checkpoints/best_model/`):
```bash
python fine_tuning/train.py
```

## Architecture

### Pipeline Flow
1. **Data** (`data/prepare_dataset.py`) — Builds a mixed dataset via synthetic code-switching: loads Tamil segments from IndicVoices-R and English segments from LibriSpeech, concatenates Tamil+silence+English pairs to create code-switched samples, resamples to 16kHz, caps segments at 8s, tags each as `monolingual_tamil` / `monolingual_english` / `code_switched`, stratified 80/10/10 split.

2. **Evaluation** (`evaluation/baseline_eval.py` + `evaluation/metrics.py`) — Loads all three baseline models, transcribes the test set, computes WER/CER stratified by segment type, and categorizes failures into 5 types: `SUBSTITUTION_SWITCH`, `DELETION_PROPER_NOUN`, `SUBSTITUTION_NUMBER`, `LANGUAGE_CONFUSION`, `INSERTION_FILLER`.

3. **Fine-tuning** (`fine_tuning/train.py`) — Applies LoRA adapters to Whisper-small (`q_proj`, `v_proj`; r=32, alpha=64), uses a custom weighted sampler that oversamples code-switched 3×, high-switch-point samples 2×, and undersamples monolingual to 50%. Trains via HuggingFace `Seq2SeqTrainer` with FP16, AdamW 8-bit, and WandB logging.

### Key Design Decisions
- **Streaming dataset loading** — IndicVoices is large; `load_indicvoices_tamil()` uses `streaming=True` to avoid downloading the full corpus.
- **Segment-type stratification** — Both the train/val/test split and the training sampler are stratified by segment type to ensure code-switched samples are represented despite being a minority class.
- **LoRA on attention only** — Only `q_proj`/`v_proj` are adapted; encoder, decoder embeddings, and LM head are frozen, keeping trainable parameters small.
- **Failure taxonomy** — The 5-category failure taxonomy in `metrics.py` directly motivates the oversampling strategy in `train.py`.

### Additional Modules
- **`analysis/report.py`** — Reads `results/baseline_wer_all.json`, computes derived metrics (CS penalty, dominant failure, shared failures), and writes `results/failure_analysis_report.md` + `results/failure_analysis_summary.json`. Run with `python analysis/report.py`.
- **`api/app.py`** — FastAPI inference server with `/health`, `/transcribe`, `/analyze`, `/model/info` endpoints. Loads fine-tuned LoRA checkpoint if available, falls back to base Whisper-small. Run with `uvicorn api.app:app --host 0.0.0.0 --port 8000`.
- **`notebooks/colab_finetune.ipynb`** — Colab-ready notebook for end-to-end fine-tuning on GPU.

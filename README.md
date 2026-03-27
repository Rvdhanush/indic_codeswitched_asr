# Code-Switched Indic ASR: Failure Analysis & Targeted Fine-tuning

## Problem Statement

Existing Indic ASR models (IndicWhisper, IndicConformer, Whisper)
perform well on clean monolingual speech but fail significantly on
code-switched speech — sentences where Tamil and English are mixed
mid-conversation (Tanglish). This is a critical gap because
real-world Indian speech, especially in urban and tech contexts,
is predominantly code-switched.

The core research question:
"Where exactly do state-of-the-art ASR models fail on Tamil-English
code-switched speech, and can targeted fine-tuning fix those
specific failure categories?"

## Why This Matters

- 130M+ English speakers in India use code-switched speech daily
- Existing benchmarks (Vistaar, IndicSUPERB) test monolingual speech
- No public failure taxonomy exists for code-switched Indic ASR
- Production systems (voice bots, transcription tools) break on
  Tanglish input — a directly observable problem from real STT
  pipeline work

## Research Contribution

1. First structured failure taxonomy for Tamil-English ASR
2. Comparative WER analysis across 3 models on code-switched speech
3. Targeted LoRA fine-tuning strategy based on failure categories
4. Published fine-tuned model + benchmark results on HuggingFace

## Architecture

```
SPRINGLab/IndicVoices-R_Tamil  +  librispeech_asr/clean
        │
        ▼
data/prepare_dataset.py
  • Resample to 16kHz mono, trim segments to 2–8s
  • Synthetic code-switching: Tamil + 0.1s silence + English
  • Tag: monolingual_tamil | monolingual_english | code_switched
  • Target mix: 40% CS, 35% Tamil, 25% English (200 samples default)
  • Stratified 80/10/10 split
        │
        ├──────────────────────────┐
        ▼                          ▼
evaluation/baseline_eval.py   fine_tuning/train.py
  Whisper-medium                LoRA on Whisper-small
  IndicWhisper                    r=32, alpha=64
  IndicWav2Vec                    q_proj + v_proj only
        │                         Weighted sampler:
        ▼                           code_switched ×3
evaluation/metrics.py               high-switch    ×2
  WER / CER                         monolingual   ×0.5
  Stratified by segment type        │
  Failure taxonomy (5 types)        ▼
        │                     checkpoints/best_model/
        ▼
  results/
```

## Failure Taxonomy

Five failure categories identified and used to guide fine-tuning data strategy:

| Category | Description |
|---|---|
| `SUBSTITUTION_SWITCH` | Transcription error at a language switch boundary |
| `DELETION_PROPER_NOUN` | Named entity or proper noun deleted from output |
| `SUBSTITUTION_NUMBER` | Number, date, or digit sequence transcribed incorrectly |
| `LANGUAGE_CONFUSION` | Tamil word transcribed in English script or vice versa |
| `INSERTION_FILLER` | Hallucinated filler word (um, uh, like, you know) |

## Datasets

| Role | Dataset | HuggingFace ID |
|---|---|---|
| Monolingual Tamil | IndicVoices-R Tamil | `SPRINGLab/IndicVoices-R_Tamil` |
| Monolingual English | LibriSpeech clean | `librispeech_asr` (clean/train.100) |
| Code-switched | Synthetic (Tamil+English) | constructed in `data/prepare_dataset.py` |

> **Why synthetic?** Public Tamil-English code-switched ASR datasets (e.g. MUCS 2021) are
> not available on HuggingFace. Real Tanglish corpora (IndicVoices, FLEURS) transcribe
> English loanwords in Tamil script, making language detection impossible at the text level.
> Synthetic concatenation produces ground-truth mixed transcripts and a real language switch
> point in the audio.

## Models Evaluated

| Model | HuggingFace ID |
|---|---|
| Whisper | `openai/whisper-medium` |
| IndicWhisper | `parthiv11/indic_whisper_nodcil` |
| IndicWav2Vec | `ai4bharat/indicwav2vec` |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add HF_TOKEN and WANDB_API_KEY to .env
```

## Usage

**1. Prepare dataset**
```bash
python data/prepare_dataset.py
# Output: data/processed/{train,validation,test}_metadata.json
```

**2. Run baseline evaluation**
```bash
python evaluation/baseline_eval.py
# Output: results/{whisper_medium,indic_whisper,indicwav2vec}_wer.json
#         results/baseline_wer_all.json
```

**3. Fine-tune**
```bash
python fine_tuning/train.py
# Config: fine_tuning/config.yaml
# Output: checkpoints/best_model/
# Logs: Weights & Biases (requires WANDB_API_KEY)
```

## Fine-tuning Configuration

Key hyperparameters (`fine_tuning/config.yaml`):

- **Base model:** `openai/whisper-small`
- **LoRA:** r=32, alpha=64, dropout=0.05, targets `q_proj` and `v_proj`
- **Training:** 3 epochs, batch size 4, gradient accumulation 4 steps, lr=1e-3, FP16, AdamW 8-bit
- **Data sampling:** code-switched ×3, high-switch-point ×2, monolingual ×0.5
- **Early stopping:** patience=3, metric=WER (lower is better)

## Repository Structure

```
data/               Dataset download, preprocessing, and split logic
evaluation/         Baseline model evaluation and failure analysis metrics
fine_tuning/        LoRA fine-tuning script and config
analysis/           Failure taxonomy reports (planned)
api/                FastAPI inference endpoint (planned)
notebooks/          Colab-compatible training notebooks (planned)
results/            WER results and comparison outputs
```

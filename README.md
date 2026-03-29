# Tamil-English Code-Switched ASR: Failure Analysis & Targeted Fine-tuning

> Structured failure taxonomy for Tanglish ASR + LoRA fine-tuning that achieves
> **41% WER reduction on code-switched speech** using only 1.44% of model parameters.

---

## Problem Statement

Real-world Indian speech — particularly in urban, tech, and professional contexts — is
predominantly code-switched: Tamil and English mixed mid-sentence (Tanglish). Existing
ASR models are trained and benchmarked exclusively on clean monolingual speech, creating
a critical production gap. Voice bots, transcription tools, and meeting assistants break
on Tanglish input in ways that are systematic and diagnosable.

**Core research question:** Where exactly do state-of-the-art models fail on Tamil-English
code-switched speech, and can targeted fine-tuning fix those specific failure categories?

---

## Key Findings

**1. All baseline models collapse on code-switched input.**
Whisper-small hallucinated repetition loops ("பிரிந்து" × 25) when encountering English
words mid-sentence. Wav2Vec2-tamil achieved WER > 1.0 on code-switched segments. The
best pre-trained baseline (Whisper-tamil-medium) still reached only 0.879 CS-WER.

**2. Targeted oversampling with LoRA outperforms all baselines using 1.44% of parameters.**
Fine-tuning only `q_proj` and `v_proj` attention layers with a weighted sampler (code-switched
×3, high-switch-point ×2, monolingual ×0.5) reduced code-switched WER from 0.964 to 0.564
— a 41% relative improvement — while beating Whisper-tamil-medium (a larger, Tamil-specialized
model) by 36%.

**3. LANGUAGE_CONFUSION and SUBSTITUTION_SWITCH are architectural, not model-specific.**
Both failure categories appear as the top-2 failures across all three baseline models.
They represent blind spots in how seq2seq ASR decoders handle language switches, not
bugs in any individual model. Fine-tuning reduced but did not eliminate them.

---

## Results

### WER by Segment Type

| Model | Overall WER | Mono-Tamil | Mono-English | Code-Switched | CS Penalty |
|---|---|---|---|---|---|
| Whisper-small (baseline) | 0.976 | 0.957 | 1.009 | 0.964 | 0.98× |
| Whisper-tamil-medium | 0.829 | 0.688 | 0.980 | 0.879 | 1.05× |
| Wav2Vec2-tamil | 1.013 | 1.031 | 1.000 | 0.999 | 0.98× |
| **Whisper-small + LoRA (ours)** | **0.682** | **0.769** | **0.566** | **0.564** | **0.84×** |

> **CS Penalty** = code-switched WER ÷ average monolingual WER. Our fine-tuned model
> scores 0.84× — meaning it handles code-switched speech *better* than monolingual speech,
> the opposite of every baseline.

### Failure Category Breakdown

| Category | Whisper-small | Whisper-tamil | Wav2Vec2-tamil | Ours (LoRA) |
|---|---|---|---|---|
| `SUBSTITUTION_SWITCH` | 46% | 46% | 64% | 58% |
| `LANGUAGE_CONFUSION` | 54% | 54% | 36% | 41% |
| `DELETION_PROPER_NOUN` | 0% | 0% | 0% | 0% |
| `SUBSTITUTION_NUMBER` | 0% | 0% | 0% | 0% |
| `INSERTION_FILLER` | 0% | 0% | 0% | 1% |

---

## Live Demo

The fine-tuned model is served via a FastAPI endpoint with Swagger UI.

```bash
git clone https://github.com/Rvdhanush/indic_codeswitched_asr
cd indic_codeswitched_asr
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/transcribe` | POST | Transcribe audio with fine-tuned model |
| `/compare` | POST | Side-by-side: baseline vs fine-tuned on same audio |
| `/analyze` | POST | Transcribe + WER + failure category (requires reference) |
| `/model/info` | GET | Loaded model metadata |

The `/compare` endpoint is the key demo — upload any Tanglish audio and see the
difference between baseline hallucination and fine-tuned output instantly.

---

## Model

The fine-tuned LoRA adapter (14MB) is published on HuggingFace:

**[Dhanush66-rv/whisper-small-tanglish-lora](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora)**

- Base: `openai/whisper-small`
- Adapter: LoRA r=32, alpha=64, targets `q_proj` + `v_proj`
- Trainable parameters: 3,538,944 / 245,273,856 (1.44%)
- Training: 5 epochs, 1786 samples (after oversampling), Google Colab T4 GPU

---

## Architecture

```
SPRINGLab/IndicVoices-R_Tamil  +  librispeech_asr/clean
        │
        ▼
data/prepare_dataset.py
  • Resample to 16kHz mono, trim segments to 2–8s
  • Synthetic code-switching: Tamil + 0.1s silence + English
  • Tag: monolingual_tamil | monolingual_english | code_switched
  • Target mix: 40% CS, 35% Tamil, 25% English
  • Stratified 80/10/10 split
        │
        ├──────────────────────────┐
        ▼                          ▼
evaluation/baseline_eval.py   fine_tuning/train.py
  3 pre-trained models           LoRA on Whisper-small
  evaluated on test set            r=32, alpha=64
        │                          q_proj + v_proj only
        ▼                          Weighted sampler:
evaluation/metrics.py               code_switched ×3
  WER / CER                         high-switch    ×2
  Stratified by segment type        monolingual   ×0.5
  Failure taxonomy (5 types)        │
        │                           ▼
        ▼                     checkpoints/best_model/
  results/
```

## Failure Taxonomy

| Category | Description |
|---|---|
| `SUBSTITUTION_SWITCH` | Transcription error at a language switch boundary |
| `DELETION_PROPER_NOUN` | Named entity or proper noun deleted from output |
| `SUBSTITUTION_NUMBER` | Number, date, or digit sequence transcribed incorrectly |
| `LANGUAGE_CONFUSION` | Tamil word transcribed in English script or vice versa |
| `INSERTION_FILLER` | Hallucinated filler word inserted into output |

---

## Datasets

| Role | Dataset | HuggingFace ID |
|---|---|---|
| Monolingual Tamil | IndicVoices-R Tamil | `SPRINGLab/IndicVoices-R_Tamil` |
| Monolingual English | LibriSpeech clean | `librispeech_asr` (clean/train.100) |
| Code-switched | Synthetic (Tamil+English concatenation) | `data/prepare_dataset.py` |

> **Why synthetic?** Public Tamil-English code-switched ASR datasets (MUCS 2021) are not
> available on HuggingFace. Real Tanglish corpora transcribe English loanwords in Tamil
> script, making language-level labelling impossible. Synthetic concatenation produces
> ground-truth mixed transcripts with a real acoustic switch point.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add HF_TOKEN and WANDB_API_KEY to .env
```

## Reproduce

```bash
# 1. Prepare dataset (streams from HuggingFace, no full download)
python data/prepare_dataset.py

# 2. Baseline evaluation
python evaluation/baseline_eval.py

# 3. Fine-tune (recommended on Colab T4+)
python fine_tuning/train.py
# or use notebooks/colab_finetune.ipynb

# 4. Failure analysis report
python analysis/report.py
```

## Fine-tuning Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `openai/whisper-small` |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 5 |
| Batch size | 4 (×4 grad accumulation) |
| Learning rate | 1e-3 with warmup |
| Optimizer | AdamW 8-bit |
| Precision | FP16 |
| Early stopping | patience=3, metric=WER |

---

## Repository Structure

```
data/               Dataset download, preprocessing, and split logic
evaluation/         Baseline model evaluation and failure analysis metrics
fine_tuning/        LoRA fine-tuning script and config
analysis/           Failure taxonomy reports and comparison summaries
api/                FastAPI inference endpoint with /compare demo
notebooks/          Colab fine-tuning notebook
results/            WER results, failure analysis, findings summary
```

---

## Citation

```bibtex
@misc{dhanush2025tanglishasr,
  title   = {Tamil-English Code-Switched ASR: Failure Analysis and Targeted LoRA Fine-tuning},
  author  = {Dhanush, R V},
  year    = {2025},
  url     = {https://github.com/Rvdhanush/indic_codeswitched_asr},
  note    = {Fine-tuned model: https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora}
}
```

# Tamil-English Code-Switched ASR: Failure Analysis & Targeted Fine-tuning

> A failure taxonomy for Tanglish ASR, a synthetic code-switched training pipeline, and a
> 14 MB LoRA adapter for Whisper-small that trains 1.44% of parameters.

### **[→ rvdhanush.github.io/indic_codeswitched_asr](https://rvdhanush.github.io/indic_codeswitched_asr/)**

[Live site](https://rvdhanush.github.io/indic_codeswitched_asr/) ·
[Model on Hugging Face](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora) ·
[Results](RESULTS.md) ·
[Research journal](RESEARCH_JOURNAL.md)

> ℹ️ The live site is served from `master:/docs`, and it presents the results as they stand on
> `master` — **including the retracted 41% figure**. `master` is the published snapshot;
> this branch (`dev`) is the corrected work in progress. Pushing `dev` does not change the site.

> ⚠️ **The headline WER comparison in earlier versions of this README was not valid.**
> Baselines and the fine-tuned model were scored on **different test sets**, so the previously
> claimed "41% WER reduction" is not supported by the data that produced it. The numbers below
> are reproduced as-measured, with the caveats stated. A frozen shared test set and corrected
> metrics are in progress — see [Known Limitations](#known-limitations).

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

**1. Whisper-small exhibits repetition collapse on out-of-distribution mixed-language input.**
It hallucinated "பிரிந்து" × 25 when encountering English words mid-sentence. This is a real,
reproducible failure mode and the single clearest qualitative result here. The *absolute* WER
figures around it are inflated by the script-mismatch problem below and should not be read as
"the model cannot hear Tamil."

**2. Targeted oversampling with LoRA reduced measured code-switched WER (0.964 → 0.564).**
Fine-tuning only `q_proj` and `v_proj` with a weighted sampler (code-switched ×3,
high-switch-point ×6, monolingual ×0.5). **This is not a like-for-like comparison** — the two
numbers come from different test sets. Also note the training data's "code-switching" is a
Tamil clip concatenated to an English clip from a *different corpus*, so part of the gain may
be the model learning the channel change rather than code-switching. Treat as provisional.

**3. The dominant reported failure category is an artifact of how it was measured.**
`LANGUAGE_CONFUSION` is the fallback branch of `categorize_failure()` — it is what gets
returned when no other rule matches, so "54% LANGUAGE_CONFUSION" means "54% uncategorized".
`DELETION_PROPER_NOUN` and `SUBSTITUTION_NUMBER` read 0% across every model because they can
never fire (the reference is lowercased before a `isupper()` check; neither corpus emits digits).
The taxonomy is effectively two live categories, not five.

---

## Results

### WER by Segment Type

| Model | Overall WER | Mono-Tamil | Mono-English | Code-Switched | CS Penalty |
|---|---|---|---|---|---|
| Whisper-small (baseline) | 0.976 | 0.957 | 1.009 | 0.964 | 0.98× |
| Whisper-tamil-medium | 0.829 | 0.688 | 0.980 | 0.879 | 1.05× |
| Wav2Vec2-tamil | 1.013 | 1.031 | 1.000 | 0.999 | 0.98× |
| **Whisper-small + LoRA (ours)** | **0.682** | **0.769** | **0.566** | **0.564** | **0.84×** |

> ⚠️ **Rows are not directly comparable.** The three baselines were scored on 50 samples from a
> 300-sample dataset build; the LoRA row on 150 samples from a 1500-sample build. Different
> builds produce different test sets, so the last row and the first three describe different data.
>
> **CS Penalty** = code-switched WER ÷ average monolingual WER. The 0.84× figure was previously
> presented as evidence that the fine-tuned model handles code-switched speech *better* than
> monolingual. That reading is probably wrong: WER here is averaged per-utterance rather than
> computed corpus-level, and code-switched samples are roughly twice as long and contain the
> easier English half, so a sub-1.0 ratio is what the arithmetic predicts regardless of model
> quality. Expect this number to move once corpus-level WER lands.

### Failure Category Breakdown

| Category | Whisper-small | Whisper-tamil | Wav2Vec2-tamil | Ours (LoRA) |
|---|---|---|---|---|
| `SUBSTITUTION_SWITCH` | 46% | 46% | 64% | 58% |
| `LANGUAGE_CONFUSION` | 54% | 54% | 36% | 41% |
| `DELETION_PROPER_NOUN` | 0% | 0% | 0% | 0% |
| `SUBSTITUTION_NUMBER` | 0% | 0% | 0% | 0% |
| `INSERTION_FILLER` | 0% | 0% | 0% | 1% |

> ⚠️ The bottom three rows are 0% because those rules are unreachable, not because those
> failures never occur. `LANGUAGE_CONFUSION` is the uncategorized fallback. See
> [Known Limitations](#known-limitations).

---

## Known Limitations

These are known defects in the methodology behind the numbers above. They are being fixed;
they are documented here because the results should not be read without them.

**1. Baselines and the fine-tuned model were evaluated on different test sets.**
`evaluation/baseline_eval.py` built its test split from a 300-sample dataset; the fine-tuned
model was scored on a split from a 1500-sample build. The split is stratified with a fixed seed,
but a different input pool yields a different test set. Any cross-row comparison in the results
table — including the previously headlined "41% reduction" — is therefore unsupported.

**Fixed (mechanism only; the numbers above still predate it).** Evaluation and training no
longer build a dataset. They read a frozen corpus (`data/corpus/`) in which every sample has a
content-addressed `uid` and every split a `corpus_id`. Each results file records the `corpus_id`
of the samples it was scored on, and both `run_all_baselines` and `analysis/report.py` refuse to
produce a comparison across differing ids. The table above will be replaced once every model has
been re-run against one frozen split — see [Corpus](#corpus).

**2. The synthetic "code-switching" is not code-switching.**
`_make_cs_sample()` concatenates a whole Tamil utterance, 0.1 s of silence, and a whole
LibriSpeech English utterance, then hardcodes `switch_count=1`. That is *sequential bilingual
audio*, not intra-sentential switching. Real Tanglish embeds English content words inside Tamil
morphosyntax, often with Tamil case suffixes attached ("meeting-ku vara mudiyuma"). Because the
two halves also come from different corpora with different recording channels, a model can score
well by detecting the channel change rather than by handling code-switching.
*Fix in progress: word-level splicing of English into Tamil utterances, with a channel pipeline
applied uniformly and a splice-detectability test as an acceptance gate.*

**3. Script mismatch inflates every WER number reported here.**
References write English in Latin script; the model emits Tamil script phonetically
(`ட்ராஃபிக்` for "traffic"). A correctly-heard word scores as fully wrong. This means
whisper-small's 0.957 WER on *monolingual Tamil* is largely measuring orthography and
normalization, not recognition.
*Fix in progress: Script-Normalized WER (SN-WER) reported alongside raw WER, plus a
`script_penalty` metric isolating how much reported error is rendering rather than mishearing.*

**4. WER is averaged per-utterance instead of corpus-level.**
`compute_stratified_wer()` computes `sum(per_utterance_wer) / n`, which over-weights short
utterances. Standard WER is total edit distance over total reference words.
*Fix in progress: corpus-level WER, with the old macro value retained for comparison.*

**5. The failure taxonomy has two live categories, not five.**
`LANGUAGE_CONFUSION` is the default `return` when no other rule matches.
`DELETION_PROPER_NOUN` is unreachable (`analyze_failures` lowercases the reference, then
`categorize_failure` tests `w[0].isupper()`). `SUBSTITUTION_NUMBER` never fires because both
source corpora spell numbers as words. Categorization also compares words by position with no
alignment, so a single insertion or deletion misclassifies everything after it.
*Fix in progress: alignment-driven categories (`SCRIPT_MISMATCH`, `SWITCH_BOUNDARY`,
`HALLUCINATION_LOOP`, `OTHER`); the unreachable categories are being removed rather than
repaired.*

---

## Live Demo

**Project site: [rvdhanush.github.io/indic_codeswitched_asr](https://rvdhanush.github.io/indic_codeswitched_asr/)**
— the waveform of a real code-switched utterance, the baseline repetition-collapse failure beside
the fine-tuned output, and the WER breakdown. Source in [`docs/`](docs/), deployed from `dev`.

Hosted inference is not up yet. To run it locally, the fine-tuned model is served via a FastAPI
endpoint with Swagger UI.

```bash
git clone https://github.com/Rvdhanush/indic_codeswitched_asr
cd indic_codeswitched_asr
pip install -e ".[serve]"
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

> The `[serve]` extra installs `python-multipart`, which FastAPI requires for the file-upload
> endpoints, and pulls CPU-only torch. Note that `checkpoints/` is gitignored — a fresh clone has
> no local adapter and the server currently falls back to base Whisper-small. Pull the adapter
> from [the Hub](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora) or train your own.

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
data/prepare_dataset.py            (builder — run once, via data.corpus)
  • Resample to 16kHz mono, trim segments to 2–8s
  • Synthetic code-switching: Tamil + 0.1s silence + English
  • Tag: monolingual_tamil | monolingual_english | code_switched
  • Target mix: 40% CS, 35% Tamil, 25% English
  • Stratified 80/10/10 split
        │
        ▼
data/corpus.py  →  data/corpus/           (frozen, content-addressed)
  • uid      = sha256(pcm16 bytes + transcript)
  • corpus_id = sha256(sorted uids of a split)
  • manifest.json + splits/*.json in git; audio/ gitignored, hash-verified
        │
        ├──────────────────────────┐
        ▼                          ▼
evaluation/baseline_eval.py   fine_tuning/train.py
  3 pre-trained models           LoRA on Whisper-small
  evaluated on test set            r=32, alpha=64
        │                          q_proj + v_proj only
        ▼                          Weighted sampler:
evaluation/metrics.py               code_switched ×3
  WER / CER                         high-switch    ×6
  Stratified by segment type        monolingual   ×0.5
  Failure taxonomy (2 live types)   │
        │                           ▼
        ▼                     checkpoints/best_model/
  results/
```

## Failure Taxonomy

| Category | Description | Status |
|---|---|---|
| `SUBSTITUTION_SWITCH` | Transcription error at a language switch boundary | live, but positional (no alignment) |
| `INSERTION_FILLER` | Hallucinated filler word inserted into output | live |
| `LANGUAGE_CONFUSION` | *Intended:* Tamil word transcribed in English script or vice versa | **fallback branch** — means "uncategorized" |
| `DELETION_PROPER_NOUN` | Named entity or proper noun deleted from output | **unreachable** (reference is lowercased first) |
| `SUBSTITUTION_NUMBER` | Number, date, or digit sequence transcribed incorrectly | **never fires** (corpora spell numbers as words) |

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

## Corpus

Evaluation and training read a **frozen corpus**, not a fresh build. This exists because the
previous pipeline rebuilt the dataset at run time and split it with `random_state=42`, which
fixes the partition of whatever list it receives — so a 300-sample pool and a 1500-sample pool
produced different test sets, and the resulting scores were tabulated as if they were comparable.

```
data/corpus/
  manifest.json            git-tracked   provenance + one record per sample
  splits/{train,validation,test}.json
                           git-tracked   {"corpus_id": ..., "uids": [...]}
  audio/<uid>.wav          gitignored    16 kHz mono PCM16
```

| Concept | Definition | Answers |
|---|---|---|
| `uid` | `sha256(pcm16 bytes + transcript)[:16]` | is this the same sample? |
| `corpus_id` | `sha256(sorted uids)[:16]` | were these models scored on the same data? |

Audio is not committed (~150 MB at 1500 samples). The manifest carries a per-file `sha256`, so a
rebuild elsewhere is verified rather than trusted:

```bash
python -m data.corpus freeze --size 1500   # build; refuses to overwrite without --force
python -m data.corpus verify               # re-hash every wav, check split disjointness
python -m data.corpus info                 # corpus_id per split, pinned dataset revisions
```

The manifest pins the HuggingFace revision of each source dataset. A rebuild that does not
reproduce the committed `corpus_id` is a finding, not a nuisance — `notebooks/colab_finetune.ipynb`
raises rather than training on a corpus that drifted.

Three checks enforce this downstream, so the original defect cannot recur silently:

- `evaluate_model` stamps `corpus_id` (the samples actually scored) and `split_corpus_id` (the
  full split they came from) into every `results/*_wer.json`.
- `run_all_baselines` leaves `baseline_wer_all.json` untouched if the models disagree.
- `analysis/report.py` prints a STOP banner and exits non-zero on mismatched or missing ids.

---

## Setup

```bash
pip install -e ".[train]"     # or ".[serve]" for API only, ".[dev]" for tests
cp .env.example .env
# Add HF_TOKEN and WANDB_API_KEY to .env
```

Dependency sets are declared in `pyproject.toml` and mirrored by `requirements/{base,serve,train,dev}.txt`.

## Reproduce

Run as modules (`python -m`), not as scripts — `python evaluation/baseline_eval.py` cannot
resolve its own package imports.

```bash
# 1. Build the frozen corpus (streams from HuggingFace, writes data/corpus/)
python -m data.corpus freeze --size 1500
python -m data.corpus verify      # re-hashes every wav; must exit 0
python -m data.corpus info        # corpus_id per split, provenance, counts

# 2. Baseline evaluation  (--dry-run lists models and the corpus, writes nothing)
python -m evaluation.baseline_eval --dry-run
python -m evaluation.baseline_eval

# 3. Fine-tune (recommended on Colab T4+)
python -m fine_tuning.train
# or use notebooks/colab_finetune.ipynb

# 4. Failure analysis report
python -m analysis.report
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
data/               Corpus builder (prepare_dataset.py) and the frozen
                    content-addressed corpus (corpus.py, corpus/)
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

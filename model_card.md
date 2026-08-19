---
language:
  - ta
  - en
license: mit
tags:
  - automatic-speech-recognition
  - code-switching
  - tamil
  - tanglish
  - whisper
  - lora
  - peft
base_model: openai/whisper-small
datasets:
  - SPRINGLab/IndicVoices-R_Tamil
  - librispeech_asr
metrics:
  - wer
---

# whisper-small-tanglish-lora

A **Whisper-small** model fine-tuned with LoRA adapters on **synthetic** Tamil-English
code-switched speech (Tanglish), built from [IndicVoices-R Tamil](https://huggingface.co/datasets/SPRINGLab/IndicVoices-R_Tamil)
and [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) English.

**Project site: [rvdhanush.github.io/indic_codeswitched_asr](https://rvdhanush.github.io/indic_codeswitched_asr/)**
· [Code](https://github.com/Rvdhanush/indic_codeswitched_asr)

> ### ⚠️ Evaluation caveat — read before citing these numbers
>
> The WER comparison below is **not like-for-like**. The baselines were scored on a 50-sample
> test split from a 300-sample dataset build; this model on a 150-sample split from a
> 1500-sample build — different data. The previously claimed "41.5% relative reduction" is
> therefore unsupported as measured.
>
> Two further caveats: WER is averaged per utterance rather than corpus-level, and the
> reference transcripts write English in Latin script while the model emits Tamil script
> phonetically, so correctly-heard English words score as fully wrong. Absolute WER here is
> inflated by an unmeasured amount.
>
> **What this model is:** a working demonstration that a 14 MB LoRA adapter measurably changes
> Whisper-small's behaviour on mixed Tamil-English audio — most visibly by eliminating the
> repetition-collapse failure. **What it is not:** a benchmarked improvement over the baselines.

## Model Description

Standard ASR models trained on monolingual data degrade significantly on code-switched speech — sentences where Tamil and English are mixed mid-utterance. This model targets that gap through **targeted fine-tuning**: training data is weighted to oversample code-switched segments and high switch-point samples, guided by a structured failure taxonomy.

| | Value |
|---|---|
| Base model | `openai/whisper-small` |
| Fine-tuning method | LoRA (PEFT) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `v_proj` |
| Training data | IndicVoices Tamil (1500 samples, stratified) |
| Languages | Tamil (`ta`), English (`en`), Tamil-English mixed |

## Intended Use

- Transcription of Tamil-English code-switched (Tanglish) speech
- Voice interfaces and STT pipelines for urban Indian users
- Research baseline for code-switched Indic ASR

**Out of scope:** Clean monolingual Tamil or English at scale — use `openai/whisper-medium` or `ai4bharat/indicwav2vec` for monolingual speech.

## Evaluation Results

WER on held-out test set (synthetic Tamil-English code-switched corpus), stratified by segment type:

| Segment Type | Whisper-small (baseline) | Whisper-tamil-medium | **This model** |
|---|---|---|---|
| Overall | 0.976 | 0.829 | **0.682** |
| Monolingual Tamil | 0.957 | 0.688 | 0.769 |
| Monolingual English | 1.009 | 0.980 | **0.566** |
| Code-switched | 0.964 | 0.879 | **0.564** |
| CS Penalty (×) | 0.98× | 1.05× | **0.84×** |

~~**41.5% relative WER reduction** on code-switched speech vs. Whisper-small baseline. **36% improvement** over the best pre-trained Tamil-specialized model.~~

**Retracted** — the columns above were measured on different test sets. See the caveat at the top.

> CS Penalty = code-switched WER ÷ average(mono-Tamil WER, mono-English WER). The 0.84× value
> was previously read as "handles code-switched better than monolingual." That reading is
> unsafe: WER is averaged per utterance, and code-switched samples are two utterances
> concatenated — one Tamil (harder) and one English (easier) — so a sub-1.0 ratio follows from
> word-count weighting regardless of model quality.

See full results in the [training repository](https://github.com/Rvdhanush/indic_codeswitched_asr).

### Failure Taxonomy

The fine-tuning strategy was derived from a structured analysis of 5 failure categories observed across all baselines:

| Category | Description | Whisper-small | Whisper-tamil | Wav2Vec2-tamil | **Ours (LoRA)** |
|---|---|---|---|---|---|
| `SUBSTITUTION_SWITCH` | Error at a Tamil↔English switch boundary | 46% | 46% | 64% | **58%** |
| `LANGUAGE_CONFUSION` | Tamil word output in English script or vice versa | 54% | 54% | 36% | **41%** |
| `DELETION_PROPER_NOUN` | Named entity deleted from output | 0% | 0% | 0% | 0% |
| `SUBSTITUTION_NUMBER` | Number or date transcribed incorrectly | 0% | 0% | 0% | 0% |
| `INSERTION_FILLER` | Hallucinated filler (um, uh, like) | 0% | 0% | 0% | 1% |

> ⚠️ Only two categories appear because the other three cannot fire. `LANGUAGE_CONFUSION` is the
> classifier's fallback branch — it means "no other rule matched", not "language confusion was
> detected". `DELETION_PROPER_NOUN` tests for capitalised words in a reference that was already
> lowercased. `SUBSTITUTION_NUMBER` looks for digits that neither source corpus produces. And
> `SUBSTITUTION_SWITCH` compares words by index with no alignment, so a single insertion or
> deletion misclassifies everything downstream. These percentages describe the measurement
> apparatus more than the models.

## Training Procedure

**Data sampling** (targeted oversampling, as actually applied to this adapter):
- Code-switched segments: ×3
- Segments with >2 language switch points: ×2 — **this branch never fired**, because synthetic
  code-switched samples hardcode `switch_count=1`. Effectively all code-switched samples got ×3.
- Monolingual segments: ×0.5 (undersampled)

**Hyperparameters:**
- Epochs: 3 (the repo's `config.yaml` default has since changed to 5)
- Batch size: 4 (effective: 16 with gradient accumulation ×4)
- Learning rate: 1e-3 with 50 warmup steps
- Optimizer: AdamW 8-bit
- Precision: FP16
- Early stopping: patience 3, metric WER

## How to Use

```python
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

base_model_id = "openai/whisper-small"
adapter_model_id = "Dhanush66-rv/whisper-small-tanglish-lora"

processor = WhisperProcessor.from_pretrained(adapter_model_id)
base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base, adapter_model_id)
model.eval()

# audio: np.ndarray, mono float32, 16kHz
def transcribe(audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(inputs, language="ta", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
```

Or via the FastAPI endpoint (see `api/app.py` in the training repo):
```bash
uvicorn api.app:app --port 8000
curl -X POST http://localhost:8000/transcribe -F "audio=@speech.wav"
```

## Limitations

**Training data is not real code-switching.** Code-switched samples are built by concatenating a
whole Tamil utterance, 0.1 s of silence, and a whole English utterance. That is *sequential
bilingual audio*, not intra-sentential switching — real Tanglish embeds English content words
inside Tamil morphosyntax, often with Tamil case suffixes attached ("meeting-ku vara mudiyuma").
The Tamil and English halves also come from different corpora with different recording channels,
so a model can score well on this data by detecting the channel change rather than by handling
code-switching. Treat any code-switching claim about this adapter as unvalidated.

**Output script.** The model emits Tamil script for English words it recognises
(`ட்ராஃபிக்` for "traffic"). Meaning is often preserved, but downstream consumers expecting
mixed-script output will need a transliteration step.

**Other limitations:**
- Trained on 1500 samples — a small corpus. Performance across speakers, accents, and domains will vary.
- Segment tagging uses `langdetect`, which is unreliable on short Tamil-script words. Per-word switch counts derived from it are close to noise.
- Not evaluated on spontaneous conversational speech; source audio is read speech.
- Never evaluated on a real Tanglish corpus. All reported numbers come from the synthetic distribution it was trained on.

## Citation

```bibtex
@misc{whisper-small-tanglish-lora,
  author    = {Dhanush, R V},
  title     = {Whisper-small fine-tuned for Tamil-English code-switched ASR},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora}
}
```

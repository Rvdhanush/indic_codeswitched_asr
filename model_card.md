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
  - ai4bharat/indicvoices
metrics:
  - wer
---

# whisper-small-tanglish-lora

A **Whisper-small** model fine-tuned with LoRA adapters on Tamil-English code-switched speech (Tanglish), trained on the [IndicVoices](https://huggingface.co/datasets/ai4bharat/indicvoices) Tamil corpus.

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

WER on IndicVoices Tamil held-out test set, stratified by segment type:

| Segment Type | Whisper-medium (baseline) | IndicWhisper (baseline) | **This model** |
|---|---|---|---|
| Overall | — | — | — |
| Monolingual Tamil | — | — | — |
| Monolingual English | — | — | — |
| Code-switched | — | — | — |
| CS Penalty (×) | — | — | — |

> Results will be populated after the evaluation run. See `results/baseline_wer_all.json` and `results/failure_analysis_report.md` in the [training repository](https://github.com/Rvdhanush/indic_codeswitched_asr).

### Failure Taxonomy

The fine-tuning strategy was derived from a structured analysis of 5 failure categories:

| Category | Description |
|---|---|
| `SUBSTITUTION_SWITCH` | Error at a Tamil↔English switch boundary |
| `DELETION_PROPER_NOUN` | Named entity deleted from output |
| `SUBSTITUTION_NUMBER` | Number or date transcribed incorrectly |
| `LANGUAGE_CONFUSION` | Tamil word output in English script or vice versa |
| `INSERTION_FILLER` | Hallucinated filler (um, uh, like) |

## Training Procedure

**Data sampling** (targeted oversampling):
- Code-switched segments: ×3
- Segments with >2 language switch points: ×2
- Monolingual segments: ×0.5 (undersampled)

**Hyperparameters:**
- Epochs: 3
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
adapter_model_id = "Rvdhanush/whisper-small-tanglish-lora"

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

- Trained on 1500 samples — a small corpus. Performance on diverse speakers, accents, and domains will vary.
- Language detection for segment tagging uses `langdetect`, which can misclassify short Tamil-script words.
- Numbers and proper nouns (especially transliterated names) remain a known weak point — see `DELETION_PROPER_NOUN` and `SUBSTITUTION_NUMBER` failure categories.
- Not evaluated on spontaneous conversational speech; training data is read-speech from IndicVoices.

## Citation

```bibtex
@misc{whisper-small-tanglish-lora,
  author    = {Rvdhanush},
  title     = {Whisper-small fine-tuned for Tamil-English code-switched ASR},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/Rvdhanush/whisper-small-tanglish-lora}
}
```

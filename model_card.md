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
  - SPRINGLab/IndicVoices-R_Tamil
  - librispeech_asr
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

WER on held-out test set (synthetic Tamil-English code-switched corpus), stratified by segment type:

| Segment Type | Whisper-small (baseline) | Whisper-tamil-medium | **This model** |
|---|---|---|---|
| Overall | 0.976 | 0.829 | **0.682** |
| Monolingual Tamil | 0.957 | 0.688 | 0.769 |
| Monolingual English | 1.009 | 0.980 | **0.566** |
| Code-switched | 0.964 | 0.879 | **0.564** |
| CS Penalty (×) | 0.98× | 1.05× | **0.84×** |

**41.5% relative WER reduction** on code-switched speech vs. Whisper-small baseline. **36% improvement** over the best pre-trained Tamil-specialized model.

> CS Penalty = code-switched WER ÷ average(mono-Tamil WER, mono-English WER). A value below 1.0 means the model handles code-switched speech *better* than monolingual speech — the opposite of all three baselines.

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

Only `SUBSTITUTION_SWITCH` and `LANGUAGE_CONFUSION` were observed — both are systemic architectural blind spots shared across all models, not model-specific bugs. Fine-tuning reduced `LANGUAGE_CONFUSION` from 54% → 41% but did not eliminate either category.

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

- Trained on 1500 samples — a small corpus. Performance on diverse speakers, accents, and domains will vary.
- Language detection for segment tagging uses `langdetect`, which can misclassify short Tamil-script words.
- Numbers and proper nouns (especially transliterated names) remain a known weak point — see `DELETION_PROPER_NOUN` and `SUBSTITUTION_NUMBER` failure categories.
- Not evaluated on spontaneous conversational speech; training data is read-speech from IndicVoices.

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

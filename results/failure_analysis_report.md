# Tamil-English Code-Switched ASR: Failure Analysis Report

Generated from `results/baseline_wer_all.json`.

## 1. WER by Segment Type

| Model          | Overall WER | Mono-Tamil WER | Mono-English WER | Code-Switched WER | CS Penalty |
| -------------- | ----------- | -------------- | ---------------- | ----------------- | ---------- |
| whisper_small  | 0.9761      | 0.9568         | 1.0089           | 0.9639            | 0.98×      |
| whisper_tamil  | 0.8294      | 0.6882         | 0.9802           | 0.8785            | 1.05×      |
| wav2vec2_tamil | 1.0134      | 1.0313         | 1.0000           | 0.9985            | 0.98×      |

> **CS Penalty** = code-switched WER ÷ average monolingual WER. A value of 2.0× means the model makes twice as many errors on code-switched speech as on clean monolingual speech.

## 2. Model Ranking on Code-Switched Speech

| Rank | Model          | Code-Switched WER |
| ---- | -------------- | ----------------- |
| 1    | whisper_tamil  | 0.8785            |
| 2    | whisper_small  | 0.9639            |
| 3    | wav2vec2_tamil | 0.9985            |

## 3. Failure Category Breakdown

| Failure Category       | Description                           | whisper_small | whisper_tamil | wav2vec2_tamil |
| ---------------------- | ------------------------------------- | ------------- | ------------- | -------------- |
| `SUBSTITUTION_SWITCH`  | Error at language switch boundary     | 23 (46%)      | 23 (46%)      | 32 (64%)       |
| `DELETION_PROPER_NOUN` | Proper noun deleted                   | 0 (0%)        | 0 (0%)        | 0 (0%)         |
| `SUBSTITUTION_NUMBER`  | Number / date transcribed incorrectly | 0 (0%)        | 0 (0%)        | 0 (0%)         |
| `LANGUAGE_CONFUSION`   | Wrong language script used            | 27 (54%)      | 27 (54%)      | 18 (36%)       |
| `INSERTION_FILLER`     | Hallucinated filler word              | 0 (0%)        | 0 (0%)        | 0 (0%)         |

**Dominant failure per model:**

- **whisper_small:** `LANGUAGE_CONFUSION` — Wrong language script used (54.0% of all failures)
- **whisper_tamil:** `LANGUAGE_CONFUSION` — Wrong language script used (54.0% of all failures)
- **wav2vec2_tamil:** `SUBSTITUTION_SWITCH` — Error at language switch boundary (64.0% of all failures)

**Systemic failures (top-2 for all models):**

- `LANGUAGE_CONFUSION` — Wrong language script used
- `SUBSTITUTION_SWITCH` — Error at language switch boundary

> These categories represent architectural blind spots shared across Whisper, IndicWhisper, and IndicWav2Vec — not model-specific bugs. They are the highest-leverage targets for fine-tuning data curation.

## 4. Fine-tuning Implications

The failure breakdown directly informs the data sampling strategy in `fine_tuning/train.py`:

| Failure Category       | Mitigation in fine_tuning/train.py                                                 |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `SUBSTITUTION_SWITCH`  | Oversample segments with high switch-point count (×2 in config)                    |
| `DELETION_PROPER_NOUN` | Include samples with named entities; avoid aggressive text normalisation           |
| `SUBSTITUTION_NUMBER`  | Ensure numeric utterances are present in training mix                              |
| `LANGUAGE_CONFUSION`   | Oversample code-switched segments overall (×3 in config)                           |
| `INSERTION_FILLER`     | Use `RemovePunctuation` + `ToLowerCase` transforms to reduce hallucination surface |

## 5. Evaluation Coverage

| Model          | Samples evaluated | Errors | Device |
| -------------- | ----------------- | ------ | ------ |
| whisper_small  | 50                | 0      | cuda   |
| whisper_tamil  | 50                | 0      | cuda   |
| wav2vec2_tamil | 50                | 0      | cuda   |

# Tamil-English Code-Switched ASR: Failure Analysis Report

Generated from `results/baseline_wer_all.json`.

> **The comparisons in this report are not currently valid.** At least one row carries no `corpus_id`, so there is no evidence the models were scored on the same test set, and for the committed results they were not. WER is also macro-averaged per utterance rather than corpus-level, and three of the five failure categories are unreachable (`LANGUAGE_CONFUSION` is the classifier fallback branch, so it means uncategorized). See README -> Known Limitations. Re-run against the frozen corpus to clear this banner:
>
> ```
> python -m data.corpus freeze
> python -m evaluation.baseline_eval
> ```

## 1. WER by Segment Type

| Model                       | Overall WER | Mono-Tamil WER | Mono-English WER | Code-Switched WER | CS Penalty |
| --------------------------- | ----------- | -------------- | ---------------- | ----------------- | ---------- |
| Whisper-small (baseline)    | 0.9761      | 0.9568         | 1.0089           | 0.9639            | 0.98×      |
| Whisper-tamil-medium        | 0.8294      | 0.6882         | 0.9802           | 0.8785            | 1.05×      |
| Wav2Vec2-tamil              | 1.0134      | 1.0313         | 1.0000           | 0.9985            | 0.98×      |
| Whisper-small + LoRA (ours) | 0.6823      | 0.7692         | 0.5663           | 0.5635            | 0.84×      |

> **CS Penalty** = code-switched WER ÷ average monolingual WER. A value of 2.0× means the model makes twice as many errors on code-switched speech as on clean monolingual speech.

## 2. Model Ranking on Code-Switched Speech

| Rank | Model                       | Code-Switched WER |
| ---- | --------------------------- | ----------------- |
| 1    | Whisper-small + LoRA (ours) | 0.5635            |
| 2    | Whisper-tamil-medium        | 0.8785            |
| 3    | Whisper-small (baseline)    | 0.9639            |
| 4    | Wav2Vec2-tamil              | 0.9985            |

## 3. Failure Category Breakdown

| Failure Category       | Description                           | Whisper-small (baseline) | Whisper-tamil-medium | Wav2Vec2-tamil | Whisper-small + LoRA (ours) |
| ---------------------- | ------------------------------------- | ------------------------ | -------------------- | -------------- | --------------------------- |
| `SUBSTITUTION_SWITCH`  | Error at language switch boundary     | 23 (46%)                 | 23 (46%)             | 32 (64%)       | 87 (58%)                    |
| `DELETION_PROPER_NOUN` | Proper noun deleted                   | 0 (0%)                   | 0 (0%)               | 0 (0%)         | 0 (0%)                      |
| `SUBSTITUTION_NUMBER`  | Number / date transcribed incorrectly | 0 (0%)                   | 0 (0%)               | 0 (0%)         | 0 (0%)                      |
| `LANGUAGE_CONFUSION`   | Wrong language script used            | 27 (54%)                 | 27 (54%)             | 18 (36%)       | 62 (41%)                    |
| `INSERTION_FILLER`     | Hallucinated filler word              | 0 (0%)                   | 0 (0%)               | 0 (0%)         | 1 (1%)                      |

**Dominant failure per model:**

- **Whisper-small (baseline):** `LANGUAGE_CONFUSION` — Wrong language script used (54.0% of all failures)
- **Whisper-tamil-medium:** `LANGUAGE_CONFUSION` — Wrong language script used (54.0% of all failures)
- **Wav2Vec2-tamil:** `SUBSTITUTION_SWITCH` — Error at language switch boundary (64.0% of all failures)
- **Whisper-small + LoRA (ours):** `SUBSTITUTION_SWITCH` — Error at language switch boundary (58.0% of all failures)

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

| Model                       | Samples evaluated | Errors | Device |
| --------------------------- | ----------------- | ------ | ------ |
| Whisper-small (baseline)    | 50                | 0      | cuda   |
| Whisper-tamil-medium        | 50                | 0      | cuda   |
| Wav2Vec2-tamil              | 50                | 0      | cuda   |
| Whisper-small + LoRA (ours) | 150               | 0      | cuda   |

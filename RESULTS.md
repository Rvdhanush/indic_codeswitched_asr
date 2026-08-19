# Results: Tamil-English Code-Switched ASR

Complete results from baseline evaluation and LoRA fine-tuning experiments.

[Live site](https://rvdhanush.github.io/indic_codeswitched_asr/) ·
[README](README.md) ·
[Research journal](RESEARCH_JOURNAL.md) ·
[Model](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora)

> ## ⚠️ Read this first
>
> **The cross-model comparisons on this page are not valid as measured.** They are preserved
> verbatim for provenance, with corrections inline. Three specific problems:
>
> 1. **Different test sets.** Baselines were scored on 50 samples from a 300-sample dataset
>    build; the LoRA model on 150 samples from a 1500-sample build. The "−41.5% relative"
>    figure compares two models on two different datasets.
> 2. **Per-utterance averaging.** WER is `sum(per_utterance_wer)/n`, not corpus-level
>    `total_errors/total_words`. This over-weights short utterances and is the likely source of
>    the sub-1.0 "CS Penalty" result.
> 3. **Script mismatch.** English in the reference is Latin script; the model emits Tamil
>    script phonetically. Correctly-heard words score as fully wrong, so every absolute WER
>    here is inflated by an unmeasured amount.
>
> See [README → Known Limitations](README.md#known-limitations). A frozen shared test set,
> corpus-level WER, and Script-Normalized WER are in progress.

---

## Full WER Comparison

### By Segment Type

| Model | Overall WER | Mono-Tamil WER | Mono-English WER | Code-Switched WER | CS Penalty |
|---|---|---|---|---|---|
| `openai/whisper-small` | 0.9761 | 0.9568 | 1.0089 | 0.9639 | 0.98× |
| `vasista22/whisper-tamil-medium` | 0.8294 | 0.6882 | 0.9802 | 0.8785 | 1.05× |
| `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250` | 1.0134 | 1.0313 | 1.0000 | 0.9985 | 0.98× |
| **`Dhanush66-rv/whisper-small-tanglish-lora`** | **0.6823** | **0.7692** | **0.5663** | **0.5635** | **0.84×** |

**CS Penalty** = code-switched WER ÷ average(mono-Tamil WER, mono-English WER).
- Value > 1.0: model is worse on code-switched than monolingual (Whisper-tamil: 1.05×)
- Value ≈ 1.0: no meaningful difference (Whisper-small: 0.98×, Wav2Vec2: 0.98×)
- Value < 1.0: model is *better* on code-switched than monolingual (**ours: 0.84×**)

### Model Ranking on Code-Switched Speech

| Rank | Model | Code-Switched WER | vs. Whisper-small baseline |
|---|---|---|---|
| 1 | Whisper-small + LoRA (ours) | 0.5635 | **−41.5% relative** |
| 2 | Whisper-tamil-medium | 0.8785 | −8.9% relative |
| 3 | Whisper-small | 0.9639 | — (baseline) |
| 4 | Wav2Vec2-tamil | 0.9985 | +3.6% relative (worse) |

> ⚠️ **The −41.5% figure is not supported.** Rank 1 was measured on a different test set from
> ranks 2–4. Additionally, the fine-tuned model's training data pairs Tamil audio from
> IndicVoices with English audio from LibriSpeech — different recording channels — so an
> unknown portion of any genuine gain may come from the model detecting the channel boundary
> rather than handling code-switching. Both issues are being addressed before a corrected
> number is published.

---

## Failure Category Breakdown

Failures were categorized using a 5-type taxonomy applied to code-switched test segments.

| Category | Description | whisper-small | whisper-tamil | wav2vec2-tamil | ours (LoRA) |
|---|---|---|---|---|---|
| `SUBSTITUTION_SWITCH` | Error at language switch boundary | 23 (46%) | 23 (46%) | 32 (64%) | 87 (58%) |
| `LANGUAGE_CONFUSION` | Wrong script for language | 27 (54%) | 27 (54%) | 18 (36%) | 62 (41%) |
| `DELETION_PROPER_NOUN` | Named entity deleted | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| `SUBSTITUTION_NUMBER` | Number/date transcribed wrong | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| `INSERTION_FILLER` | Hallucinated filler word | 0 (0%) | 0 (0%) | 0 (0%) | 1 (1%) |

**Key observation (revised):** `SUBSTITUTION_SWITCH` and `LANGUAGE_CONFUSION` are the only two
categories observed across all models — but that is a property of the classifier, not of the
models. `LANGUAGE_CONFUSION` is the fallback branch returned when no other rule matches, and
the other two categories are unreachable: `DELETION_PROPER_NOUN` tests `w[0].isupper()` on a
reference that was already lowercased, and `SUBSTITUTION_NUMBER` looks for digits that neither
corpus produces. So "LANGUAGE_CONFUSION dropped from 54% to 41%" describes a change in the
uncategorized rate, not a reduction in language confusion.

`SUBSTITUTION_SWITCH` is also unreliable: it compares `ref_words[i]` to `hyp_words[i]` by
position with no alignment, so one inserted or deleted word misclassifies everything after it.

---

## The Demo Result: Baseline vs Fine-tuned

The most concrete illustration of the difference. Input audio: a naturally spoken Tanglish
sentence — *"Tomorrow meeting ku vara mudiyuma, I'll be 10 minutes late, traffic problem iruku"*

**Baseline Whisper-small output:**
```
மாற்றும் மீட்டிங்கு வர முடியுமா? நான் பிறகு 10 நிறுத்தில் பிரிந்து பிரிந்து
பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து
பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து பிரிந்து...
```
The model hallucinated "பிரிந்து" (meaning "separated") 25+ times. This is repetition
collapse — a known seq2seq failure where the decoder locks into a high-probability local
pattern when it encounters out-of-distribution input (English words mid-sentence).

**Fine-tuned LoRA output:**
```
துமாரோம் மீட்டிங்கு வர முடியுமா ஆயில்பி டென்னுனெட்ஸ்லேட் ட்ராஃபிக் பிராவலம் இருக்கு
```
No hallucination. The full sentence is transcribed coherently. The Tamil portions
("மீட்டிங்கு வர முடியுமா", "இருக்கு") are correct. The English words are recognized
and transcribed phonetically in Tamil script — this is `LANGUAGE_CONFUSION`, the
remaining failure — but the meaning is preserved and the output is usable.

**The gap in one sentence:** baseline produces unusable garbage; fine-tuned produces
imperfect but coherent and recoverable output.

---

## CS Penalty Analysis

> ⚠️ **This section's conclusion does not survive scrutiny.** It is kept for provenance.
> The argument below treats a sub-1.0 CS Penalty as evidence that oversampling worked. But
> WER here is averaged per utterance, and code-switched samples are ~2× longer than
> monolingual ones because they are two utterances concatenated — one Tamil (hard, 0.769) and
> one English (easier, 0.566). A ratio below 1.0 is what that arithmetic produces regardless
> of whether the model learned anything about code-switching. The claim needs corpus-level WER
> on a shared test set before it can be made.

The CS Penalty metric reveals something important about how each model *relates* to
code-switching, not just how well it transcribes it.

- **Whisper-tamil-medium (1.05×):** Despite being Tamil-specialized, it is *worse* on
  code-switched than monolingual. Tamil specialization hurt — the model over-predicts Tamil
  and resists English words even more than the base Whisper-small.

- **Whisper-small and Wav2Vec2-tamil (0.98×):** Essentially no difference between
  monolingual and code-switched performance. Both are uniformly bad at everything,
  so the ratio is flat.

- **Our fine-tuned model (0.84×):** Better on code-switched than monolingual. This is
  the expected result when a model is *specifically trained* to handle code-switching —
  it has seen and practiced on that distribution. The monolingual Tamil WER (0.769) is
  higher than Whisper-tamil-medium's (0.688) because we did not specialize for monolingual
  Tamil; we specialized for switching.

~~This is not a side effect. The CS Penalty dropping below 1.0 is a direct confirmation
that the oversampling strategy worked as designed.~~

**Retracted.** See the warning at the top of this section. The sub-1.0 ratio is consistent
with a word-count weighting artifact and is not evidence about the oversampling strategy
either way.

---

## What The Numbers Mean In Plain English

**Overall WER 0.682** means the model gets roughly 1 in 3 words wrong on average across
all speech types. This sounds high but needs context: the test set includes Tamil speech
(inherently harder for any model trained primarily on English data) and synthetic
code-switched audio (the hardest distribution).

**Code-switched WER 0.564** means on the hardest input type — mixed Tamil-English
mid-sentence — the model gets about 1 in 2 words wrong. This is a 41% improvement
over a baseline that was collapsing entirely on the same input.

**The practical threshold for usability** in a voice bot or transcription tool is
typically WER < 0.3 for production quality. At 0.564 we are above that threshold —
the model is not production-ready, but it is a clear proof of concept that the
methodology works. A larger dataset, more epochs, and mixed-script training data
would be the path to sub-0.3 WER on Tanglish.

---

## Evaluation Coverage

| Model | Samples Evaluated | Errors | Device |
|---|---|---|---|
| Whisper-small | 50 | 0 | CUDA (RTX 3050) |
| Whisper-tamil-medium | 50 | 0 | CUDA (RTX 3050) |
| Wav2Vec2-tamil | 50 | 0 | CUDA (RTX 3050) |
| Whisper-small + LoRA | 150 | 0 | CUDA (RTX 3050) |

> ⚠️ **This table is the clearest statement of the problem.** The baselines and the fine-tuned
> model were not evaluated on the same data — the 50-sample and 150-sample test sets come from
> different dataset builds (300 vs 1500 samples), so they contain different audio. Every
> cross-model number on this page inherits that defect.

Baseline models evaluated on a 50-sample test split from a 300-sample build; fine-tuned model
evaluated on a 150-sample test split from a 1500-sample build.

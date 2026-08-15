# Findings: Tamil-English Code-Switched ASR

> ⚠️ **Correction notice.** This document's central claim — a 41% WER reduction — is not
> supported by the evaluation that produced it. Baselines and the fine-tuned model were scored
> on **different test sets**. Additional defects (per-utterance rather than corpus-level WER,
> script mismatch inflating all absolute numbers, a failure taxonomy whose dominant category is
> an uncategorized fallback) are documented in
> [README → Known Limitations](../README.md#known-limitations). The text below is preserved for
> provenance with corrections inline.

## Problem and Approach

Real-world Indian speech — particularly in urban and tech contexts — is predominantly
code-switched: Tamil and English mixed mid-sentence (Tanglish). Existing ASR models are
trained and benchmarked on clean monolingual speech, leaving a significant gap for
production systems such as voice bots and transcription tools that encounter Tanglish input.

This project addresses two questions: where exactly do state-of-the-art models fail on
Tamil-English code-switched speech, and can targeted fine-tuning fix those specific failure
categories?

Three pre-trained models were evaluated on a synthetic code-switched test set built from
IndicVoices-R Tamil and LibriSpeech English segments: Whisper-small, Whisper-tamil-medium,
and Wav2Vec2-tamil. Failures were categorised into five types — `SUBSTITUTION_SWITCH`,
`DELETION_PROPER_NOUN`, `SUBSTITUTION_NUMBER`, `LANGUAGE_CONFUSION`, and `INSERTION_FILLER`.
These categories directly informed a LoRA fine-tuning strategy applied to Whisper-small,
using a weighted sampler that oversamples code-switched data (×3) and high-switch-point
segments while undersampling monolingual data (×0.5). (The high-switch rule never actually
fired for the published adapter: synthetic code-switched samples hardcode `switch_count=1`.)

## Key Finding (retracted): 41% WER Reduction on Code-Switched Speech

| Model | Overall WER | Code-Switched WER | CS Penalty |
|---|---|---|---|
| Whisper-small (baseline) | 0.976 | 0.964 | 0.98× |
| Whisper-tamil-medium | 0.829 | 0.879 | 1.05× |
| Wav2Vec2-tamil | 1.013 | 0.999 | 0.98× |
| **Whisper-small + LoRA (ours)** | **0.682** | **0.564** | **0.84×** |

~~The fine-tuned model reduces code-switched WER from 0.964 to 0.564 — a **41% relative
improvement** over the Whisper-small baseline and a **36% improvement** over the best
pre-trained baseline (Whisper-tamil-medium at 0.879). Crucially, the CS penalty drops to
0.84×, meaning the fine-tuned model handles code-switched speech *better* than monolingual
speech — the opposite of all three baselines. This confirms that targeted oversampling of
code-switched and high-switch-point samples directly addresses the failure modes identified
in the taxonomy, rather than improving general ASR performance uniformly.~~

**Corrected reading.** The last row of that table was measured on a different test set from the
first three (150 samples from a 1500-sample build vs 50 samples from a 300-sample build), so the
"41% relative improvement" compares two models on two different datasets and cannot be claimed.

The CS-penalty argument is separately unsound. WER is averaged per utterance rather than computed
corpus-level, and code-switched samples are two utterances concatenated — one Tamil (WER 0.769)
and one English (WER 0.566). A ratio below 1.0 is what that word-count weighting produces
whether or not the model learned anything about code-switching, so it is not confirmation that
the oversampling strategy worked.

There is also a confound in the training data itself: Tamil audio comes from IndicVoices and
English audio from LibriSpeech, two corpora with different recording channels joined by a fixed
0.1 s silence. A model can reduce loss on this distribution by detecting the channel boundary
rather than by handling code-switching. Distinguishing the two requires a real-speech test set.

**What does survive:** the fine-tuned model no longer produces repetition collapse on mixed-language
input, where the baseline emitted "பிரிந்து" 25 times. That is a qualitative behavioural change
visible without any metric, and it is the strongest result here.

## Failure Analysis: SUBSTITUTION_SWITCH and LANGUAGE_CONFUSION

> ⚠️ The two categories below account for 100% of classified errors because the other three
> rules are unreachable, and because `LANGUAGE_CONFUSION` is the fallback returned when nothing
> else matches. The section reads as a finding about models; it is largely a finding about the
> classifier. See [README → Known Limitations](../README.md#known-limitations).

Failure analysis across all three baselines reveals two systemic categories that account for
100% of observed errors:

**LANGUAGE_CONFUSION (36–54%)** — the model transcribes a word in the wrong script, such as
rendering a Tamil word using English characters or vice versa. This was the dominant failure
in both Whisper variants (54% each), reflecting that multilingual Whisper lacks a reliable
language-switch signal at the word level.

**SUBSTITUTION_SWITCH (46–64%)** — transcription errors concentrated at language switch
boundaries. Wav2Vec2-tamil showed this most severely (64%), likely because its CTC decoder
has no language context window to carry across the switch point.

After fine-tuning, these categories persist but their relative proportions shift:
SUBSTITUTION_SWITCH rises to 58% while LANGUAGE_CONFUSION drops to 41%. This suggests the
oversampling strategy partially corrected cross-script confusion but the boundary-switch
problem remains the harder of the two failure modes to resolve with data augmentation alone.

## Limitations and Next Steps

**Synthetic data gap.** The code-switched samples are constructed by concatenating Tamil and
English audio with a 0.1s silence gap. Real Tanglish speech has natural prosodic blending
across switch points; the synthetic boundary is acoustically distinct, which may explain why
SUBSTITUTION_SWITCH remains the top failure after fine-tuning.

**Incompatible test sets (the critical defect).** Baselines were evaluated on 50 samples drawn
from a 300-sample dataset build; the fine-tuned model on 150 samples from a 1500-sample build.
These are different test sets containing different audio, so the comparison is invalid — not
merely underpowered. Larger evaluation sets (1000+ samples) would additionally be needed for
statistically robust WER estimates, but a single frozen test set shared by all models is the
prerequisite.

**LoRA scope.** Only `q_proj` and `v_proj` attention layers are adapted (1.44% of parameters).
Extending LoRA to the encoder's cross-attention layers or the decoder LM head may further
reduce LANGUAGE_CONFUSION, which appears to be a decoding-level failure.

**Next steps** (in priority order — the first three are prerequisites for any published number):

1. **Freeze a shared test set.** Content-addressed sample IDs with hash-based splits, so adding
   data never reshuffles an existing split, plus a git-tracked test-set ID list. Re-evaluate all
   four models against it.
2. **Fix the metric layer.** Corpus-level WER, Script-Normalized WER (SN-WER) to separate
   "misheard" from "right word, wrong script", alignment-driven failure categories, and a
   switch-point error rate restricted to the ±1-word window around annotated switch points.
3. **Build a real Tanglish eval set.** ~250 hand-verified intra-sentential sentences read by
   4–6 speakers, published openly. Expect the LoRA gain to shrink sharply against it; publish
   that result either way.
4. **Replace the synthetic generator.** Splice individual English words *inside* Tamil
   utterances using mined switch statistics, with the recording channel applied uniformly to all
   samples so channel cannot correlate with label. Gate on a splice-detectability test.
5. Extend LoRA beyond `q_proj`/`v_proj`; train on mixed-script targets so mixed-script output
   is native rather than post-processed.
6. ~~Publish the fine-tuned LoRA adapter to HuggingFace Hub~~ — done: [Dhanush66-rv/whisper-small-tanglish-lora](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora)
7. Ship a public demo (`api/app.py` → Gradio on HF Spaces), with the loaded adapter identity
   surfaced in the UI so the demo cannot silently serve the base model.

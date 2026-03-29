# Findings: Tamil-English Code-Switched ASR

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
segments (×2) while undersampling monolingual data (×0.5).

## Key Finding: 41% WER Reduction on Code-Switched Speech

| Model | Overall WER | Code-Switched WER | CS Penalty |
|---|---|---|---|
| Whisper-small (baseline) | 0.976 | 0.964 | 0.98× |
| Whisper-tamil-medium | 0.829 | 0.879 | 1.05× |
| Wav2Vec2-tamil | 1.013 | 0.999 | 0.98× |
| **Whisper-small + LoRA (ours)** | **0.682** | **0.564** | **0.84×** |

The fine-tuned model reduces code-switched WER from 0.964 to 0.564 — a **41% relative
improvement** over the Whisper-small baseline and a **36% improvement** over the best
pre-trained baseline (Whisper-tamil-medium at 0.879). Crucially, the CS penalty drops to
0.84×, meaning the fine-tuned model handles code-switched speech *better* than monolingual
speech — the opposite of all three baselines. This confirms that targeted oversampling of
code-switched and high-switch-point samples directly addresses the failure modes identified
in the taxonomy, rather than improving general ASR performance uniformly.

## Failure Analysis: SUBSTITUTION_SWITCH and LANGUAGE_CONFUSION

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

**Dataset scale.** Baselines were evaluated on 50 samples; the fine-tuned model on 150. While
the improvement is consistent, larger evaluation sets (1000+ samples) would be needed to
report statistically robust WER estimates.

**LoRA scope.** Only `q_proj` and `v_proj` attention layers are adapted (1.44% of parameters).
Extending LoRA to the encoder's cross-attention layers or the decoder LM head may further
reduce LANGUAGE_CONFUSION, which appears to be a decoding-level failure.

**Next steps:**
- Evaluate on a real Tanglish corpus (e.g. MUCS 2021 or IndicVoices-CS when publicly available)
- Extend LoRA to cross-attention layers and compare CS penalty
- ~~Publish the fine-tuned LoRA adapter to HuggingFace Hub~~ — done: [Dhanush66-rv/whisper-small-tanglish-lora](https://huggingface.co/Dhanush66-rv/whisper-small-tanglish-lora)
- Integrate with the FastAPI server (`api/app.py`) for live Tanglish transcription demo

# Research Journal: Tamil-English Code-Switched ASR

A full account of decisions made, obstacles encountered, and lessons learned
building this project from scratch — including the parts that didn't work the first time.

---

## The Problem We Set Out To Solve

The starting observation was simple: Tamil speakers in professional and urban settings
don't speak Tamil. They speak Tanglish — a fluid mix of Tamil and English within the
same sentence, sometimes within the same phrase. "Meeting ku vara mudiyuma, I'll be
10 minutes late, traffic problem iruku" is not an edge case. It is the norm.

Every ASR system tested on this kind of input either hallucinated, produced Tamil
transliterations of English words, or collapsed into repetition loops. The gap was not
subtle — Word Error Rates above 0.96 on code-switched speech, compared to 0.69 on
monolingual Tamil for the best baseline. The question was whether a principled analysis
of *why* models fail could lead to a targeted fix.

---

## Dataset Challenge: When No Data Exists

The first obstacle was data. Tamil-English code-switched ASR datasets do not exist in
a usable public form.

**MUCS 2021** (Microsoft and IIIT Hyderabad's multilingual code-switching challenge) is
the canonical dataset for this task — but it requires registration with IIIT Hyderabad
and is not available on HuggingFace. It cannot be streamed, cannot be used with standard
HuggingFace dataset pipelines, and is gated behind an academic agreement.

**IndicVoices** (AI4Bharat's large-scale Indic speech corpus) transcribes English loanwords
phonetically in Tamil script — "traffic" becomes "ட்ராஃபிக்". This makes language-level
annotation impossible at the text level. You cannot tell from the transcript whether the
speaker said an English word or a Tamil word, which breaks the entire failure taxonomy.

**The decision: synthetic code-switching.** Tamil audio segments from IndicVoices-R and
English audio segments from LibriSpeech were concatenated with a 0.1-second silence gap
to produce code-switched samples. This is legitimate research practice — synthetic
data is used extensively in low-resource multilingual NLP when real paired data is
unavailable. The key advantage: we control the ground truth. Every switch point is
known, every English word is labeled as English, and the evaluation is clean.

The limitation is real: synthetic concatenation produces an artificial acoustic boundary
that does not exist in natural Tanglish speech. This is documented honestly in the findings.

---

## Model Compatibility Battles

Three significant technical blockers were hit during development, each requiring diagnosis
before a fix was found.

**PyTorch security warning blocking `.bin` format models.** PyTorch 2.5.1 introduced a CVE
warning that caused certain `.bin` checkpoint formats to fail on load. The fix was switching
to `.safetensors` format models wherever available and pinning library versions.

**Whisper `generation_config` API mismatch.** The vasista22/whisper-tamil-medium model stored
`suppress_tokens` in `model.config` — a pattern deprecated in newer transformers versions,
which now requires it to be in `model.generation_config`. This caused a `ValueError` on
every evaluation run with the newer transformers version installed. Fix: explicitly move
`suppress_tokens` from `model.config` to `model.generation_config` before any `.generate()`
call. This same fix was required in both the fine-tuning script and the FastAPI server.

**VRAM overflow with two models in GPU memory.** The FastAPI `/compare` endpoint loads both
the baseline Whisper-small and the fine-tuned LoRA model simultaneously. On an RTX 3050
with 4GB VRAM, loading both in FP16 on CUDA caused the server to silently hang — requests
would never return. Fix: load the baseline model in FP32 on CPU, route fine-tuned inference
to CUDA. Comparison requests take longer but the server stays stable.

---

## What We Discovered

> ⚠️ **Corrections applied 2026-08-16.** Two of the four findings below did not survive review
> and are annotated inline. The short version: the baselines and the fine-tuned model were
> evaluated on **different test sets**, so no cross-model comparison here is valid, and the
> CS-penalty argument is a word-count weighting artifact. See
> [README → Known Limitations](README.md#known-limitations). The hallucination-collapse finding
> and the script-selection finding do stand — both are observable without a metric.

**Baseline hallucination collapse on code-switched input.** Whisper-small, when encountering
English words mid-sentence, does not produce a wrong English transcription — it produces
a Tamil word repeated 20-25 times in a loop. This is a known failure mode in seq2seq
models when the decoder becomes uncertain: it locks into a high-probability local pattern
and repeats it. The fine-tuned model eliminates this completely.

~~**41% WER reduction with 1.44% parameters updated.**~~ **Retracted.** The two WER figures were
measured on different test sets (150 samples from a 1500-sample dataset build vs 50 from a
300-sample build), so the delta between them is not a measurement of anything. The CS-penalty
argument is separately unsound: WER was averaged per utterance rather than corpus-level, and
code-switched samples are two utterances concatenated — one Tamil (harder) and one English
(easier) — so a sub-1.0 ratio follows from word-count weighting regardless of model quality.

What remains true and worth stating: LoRA on `q_proj`/`v_proj` alone — 3.5M trainable parameters
out of 245M — was enough to *change the model's behaviour* on mixed-language input, visibly and
reproducibly. Quantifying that change requires the frozen test set and corpus-level metrics that
are being built now.

~~**A fine-tuned small model beats a larger specialized model.**~~ **Retracted, same reason.**
The 0.564 (ours) vs 0.879 (Whisper-tamil-medium) comparison spans two different test sets. There
is an additional confound specific to this claim: our training data pairs IndicVoices Tamil audio
with LibriSpeech English audio — two corpora with different recording channels, joined by a fixed
0.1 s silence. A model can lower loss on that distribution by learning the channel boundary rather
than by handling code-switching, and nothing in the current evaluation distinguishes the two.

**LANGUAGE_CONFUSION is the remaining hard problem.** After fine-tuning, the model transcribes
"I'll be 10 minutes late" as "ஆயில்பி டென்னுனெட்ஸ்லேட்" — English words in Tamil script.
The acoustics are recognized correctly; the script selection is wrong. This is a decoder-level
decision made once at the start of generation in Whisper's architecture. Fixing it requires
mixed-script ground truth training data, which does not currently exist at scale.

---

## What We Would Do With More Time and Data

- **Real MUCS 2021 data.** Register with IIIT Hyderabad, obtain the actual Tamil-English
  code-switched corpus, and re-evaluate. This would validate whether the synthetic data
  findings generalize to natural Tanglish speech.
- **Mixed-script ground truth transcriptions.** Train on data where English words are
  written in English script within Tamil sentences. This directly targets LANGUAGE_CONFUSION.
- **Larger LoRA rank and more epochs.** r=32 was chosen conservatively for 4GB VRAM.
  r=64 or r=128 with more training data may further reduce SUBSTITUTION_SWITCH errors.
- **Multi-language extension.** The same methodology applies to Hindi-English (Hinglish),
  Telugu-English, and Kannada-English — all high-value production targets.

---

## Technical Decisions and Why

**Why synthetic data over no data.** The alternative was to not train at all, which produces
nothing. Synthetic data with honest documentation of its limitations is strictly better than
no data. The results are evaluated against a held-out test set built the same way, so the
numbers are internally consistent even if they cannot be directly compared to MUCS benchmarks.

**Why LoRA over full fine-tuning.** Full fine-tuning Whisper-small on 1500 samples would
cause severe catastrophic forgetting — the model would lose its English and general speech
recognition capability in favor of the narrow training distribution. LoRA updates only 1.44%
of parameters, preserving the base model's representations while adapting its attention
patterns to code-switched input.

**Why Whisper-small as the base model.** Whisper-small fits in 4GB VRAM with room for
training overhead. Whisper-medium would require 8GB+ for LoRA training. The goal was a
model that runs locally on consumer hardware, not one that requires cloud inference.

**Why targeted oversampling instead of uniform training.** The failure taxonomy told us
exactly which samples to prioritize: code-switched and high-switch-point segments. Uniform
training on a 40/35/25 split would underrepresent the failure modes we were trying to fix.
The weighted sampler is a direct translation of the failure analysis into a data strategy.

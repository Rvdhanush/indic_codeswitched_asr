import jiwer
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0

@dataclass
class FailureCase:
    reference: str
    hypothesis: str
    failure_type: str
    position: int
    context: str

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])
    return jiwer.wer(
        reference, hypothesis,
        reference_transform=transformation,
        hypothesis_transform=transformation
    )

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    return jiwer.cer(reference, hypothesis)

def detect_language_composition(text: str) -> Dict[str, float]:
    """Detect language mix in text. Returns dict of lang->probability."""
    try:
        langs = detect_langs(text)
        return {str(l.lang): round(l.prob, 3) for l in langs}
    except Exception:
        return {"unknown": 1.0}

def tag_segment_type(transcript: str) -> str:
    """
    Tag a transcript segment as:
    - monolingual_tamil
    - monolingual_english
    - code_switched
    """
    lang_comp = detect_language_composition(transcript)
    en_prob = lang_comp.get("en", 0.0)
    ta_prob = lang_comp.get("ta", 0.0)

    if en_prob > 0.9:
        return "monolingual_english"
    elif ta_prob > 0.9:
        return "monolingual_tamil"
    else:
        return "code_switched"

def find_switch_boundaries(words: List[str]) -> List[int]:
    """
    Find indices where language switches occur in a word list.
    Returns list of positions where switch happens.
    """
    boundaries = []
    prev_lang = None
    for i, word in enumerate(words):
        try:
            langs = detect_langs(word)
            curr_lang = langs[0].lang if langs else "unknown"
        except Exception:
            curr_lang = "unknown"
        if prev_lang and curr_lang != prev_lang and curr_lang != "unknown":
            boundaries.append(i)
        prev_lang = curr_lang
    return boundaries

def categorize_failure(
    ref_words: List[str],
    hyp_words: List[str],
    switch_boundaries: List[int]
) -> str:
    """
    Categorize a transcription failure into one of 5 types:
    - SUBSTITUTION_SWITCH: error at language switch boundary
    - DELETION_PROPER_NOUN: proper noun was deleted
    - SUBSTITUTION_NUMBER: number/date error
    - LANGUAGE_CONFUSION: Tamil word transcribed as English or vice versa
    - INSERTION_FILLER: hallucinated filler word
    """
    ref_text = " ".join(ref_words)
    hyp_text = " ".join(hyp_words)

    # Check for filler insertions
    fillers = ["um", "uh", "ah", "er", "like", "you know"]
    for filler in fillers:
        if filler in hyp_words and filler not in ref_words:
            return "INSERTION_FILLER"

    # Check for number errors
    number_pattern = re.compile(r'\b\d+\b')
    if number_pattern.search(ref_text) and not number_pattern.search(hyp_text):
        return "SUBSTITUTION_NUMBER"

    # Check for proper noun deletion (capitalized words in reference)
    proper_nouns = [w for w in ref_words if w and w[0].isupper()]
    for noun in proper_nouns:
        if noun not in hyp_words:
            return "DELETION_PROPER_NOUN"

    # Check if error is near switch boundary
    if switch_boundaries:
        for boundary in switch_boundaries:
            window = range(max(0, boundary-2), min(len(ref_words), boundary+2))
            for i in window:
                if i < len(ref_words) and i < len(hyp_words):
                    if ref_words[i] != hyp_words[i]:
                        return "SUBSTITUTION_SWITCH"

    # Default: language confusion
    return "LANGUAGE_CONFUSION"

def analyze_failures(
    reference: str,
    hypothesis: str
) -> Dict:
    """
    Full failure analysis for a single reference/hypothesis pair.
    Returns structured failure report.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    segment_type = tag_segment_type(reference)
    switch_boundaries = find_switch_boundaries(ref_words)
    failure_type = categorize_failure(ref_words, hyp_words, switch_boundaries)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "segment_type": segment_type,
        "switch_boundary_count": len(switch_boundaries),
        "failure_type": failure_type,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words)
    }

def compute_stratified_wer(results: List[Dict]) -> Dict:
    """
    Compute WER broken down by segment type.
    Input: list of analyze_failures() outputs with added wer field.
    """
    groups = {
        "overall": [],
        "monolingual_tamil": [],
        "monolingual_english": [],
        "code_switched": []
    }
    failure_counts = {
        "SUBSTITUTION_SWITCH": 0,
        "DELETION_PROPER_NOUN": 0,
        "SUBSTITUTION_NUMBER": 0,
        "LANGUAGE_CONFUSION": 0,
        "INSERTION_FILLER": 0
    }

    for r in results:
        groups["overall"].append(r["wer"])
        seg_type = r.get("segment_type", "unknown")
        if seg_type in groups:
            groups[seg_type].append(r["wer"])
        ft = r.get("failure_type")
        if ft in failure_counts:
            failure_counts[ft] += 1

    stratified = {}
    for key, wer_list in groups.items():
        if wer_list:
            stratified[f"{key}_wer"] = round(
                sum(wer_list) / len(wer_list), 4
            )
        else:
            stratified[f"{key}_wer"] = None

    stratified["failure_breakdown"] = failure_counts
    stratified["total_samples"] = len(results)
    return stratified

"""
Extracts diagnosis from doctor's utterances.
Identifies diagnostic language patterns in Hindi medical consultations.
"""

from typing import List, Dict, Any

# Single keywords that indicate diagnostic statements
DIAGNOSIS_CUES = {
    "निदान",
    "डायग्नोसिस", 
    "diagnosis",
    "रोग",
    "बीमारी",
    "बिमारी",  # Common misspelling
    "समस्या",
    "इन्फेक्शन",
    "infection",
}

# Multi-word phrases that indicate diagnostic reasoning
# Format: tuple of words that must ALL be present in the sentence
DIAGNOSIS_PHRASES = [
    # Case-style
    ("केस", "है"),              # okay: "यह बुखार का केस है"
    ("केस", "लग"),             # "यह इन्फेक्शन का केस लग रहा है"

    # Patient-directed
    ("आपको", "हुआ", "है"),     # "आपको वायरल बुखार हुआ है"
    ("आपको", "इन्फेक्शन", "है"),
    ("आपको", "बीमारी", "है"),
    ("आपको", "बिमारी", "है"),

    # Disease words with copula
    ("रोग", "है"),
    ("इन्फेक्शन", "है"),

    # Probabilistic (still usually diagnostic)
    ("हो", "सकता", "है"),      # "वायरल इन्फेक्शन हो सकता है"
    ("लगता", "है"),            # use with caution
]


def contains_diagnosis_signal(text: str) -> bool:
    """
    Check if text contains diagnostic language.

    Returns True if:
    1. Any single-word diagnostic cue is present, OR
    2. All words from any multi-word phrase are present
    """
    # Check single-word cues
    if any(cue in text for cue in DIAGNOSIS_CUES):
        return True

    # Check multi-word phrases (all words must be present)
    for phrase_words in DIAGNOSIS_PHRASES:
        if all(word in text for word in phrase_words):
            return True

    return False

def extract_doctor_diagnosis(entries: List[Dict[str, str]]) -> List[str]:
    """
    Extract diagnosis statements from doctor's utterances.

    Strategy: 
    1. Accumulate doctor's consecutive sentences into a buffer
    2. When diagnostic keywords appear, capture the full buffer as one diagnosis
    3. Reset buffer when patient speaks (context boundary)

    Args:
        entries: List of transcript entries with "speaker" and "text" keys

    Returns:
        List of diagnosis strings

    Examples:
        >>> entries = [
        ...     {"speaker": "doctor", "text": "यह बुखार का केस है"},
        ...     {"speaker": "doctor", "text": "आपको वायरल इन्फेक्शन हुआ है"}
        ... ]
        >>> extract_doctor_diagnosis(entries)
        ['यह बुखार का केस है', 'आपको वायरल इन्फेक्शन हुआ है']
    """
    diagnoses: List[str] = []
    buffer = ""

    for entry in entries:
        # Reset buffer when patient speaks (context boundary)
        if entry.get("speaker") != "doctor":
            buffer = ""
            continue

        # Accumulate doctor's consecutive sentences
        text = entry.get("text", "").strip()
        buffer = f"{buffer} {text}".strip()

        # Check if diagnostic signal appears
        if contains_diagnosis_signal(buffer):
            diagnoses.append(buffer)
            buffer = ""  # Reset after capturing diagnosis

    return diagnoses

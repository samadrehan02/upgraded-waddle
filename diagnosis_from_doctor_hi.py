"""
Extracts diagnosis from doctor's utterances.
Identifies diagnostic language patterns in Hindi medical consultations.
"""
from typing import List, Dict, Any

# Keywords that indicate diagnostic statements
DIAGNOSIS_CUES = {
    "निदान",
    "डायग्नोसिस",
    "diagnosis",
    "रोग",
}

# Phrases that indicate diagnostic reasoning
DIAGNOSIS_PHRASES = [
    ("केस", "लग"),      # FIXED: was ("केस", "लगता"), now matches "लग रहा है"
    ("केस", "है"),      # NEW: "case है"
    ("हो", "सकता"),     # NEW: "might be"
]


def contains_diagnosis_signal(text: str) -> bool:
    """Check if text contains diagnostic language."""
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
    
    Strategy: Accumulate doctor's consecutive sentences into a buffer.
    When diagnostic keywords appear, capture the full buffer as one diagnosis.
    
    Args:
        entries: List of transcript entries with "speaker" and "text" keys
        
    Returns:
        List of diagnosis strings
    """
    diagnoses: List[str] = []
    buffer = ""
    
    for entry in entries:
        # Reset buffer when patient speaks
        if entry.get("speaker") != "doctor":
            buffer = ""
            continue
        
        # Accumulate doctor's consecutive sentences
        text = entry.get("text", "").strip()
        buffer = f"{buffer} {text}".strip()
        
        # Check if diagnostic signal appears
        if contains_diagnosis_signal(buffer):
            diagnoses.append(buffer)
            buffer = ""
    
    return diagnoses

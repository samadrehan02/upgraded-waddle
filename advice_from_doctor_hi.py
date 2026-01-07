"""
Extracts medical advice from doctor's utterances.
Identifies advisory language patterns in Hindi medical consultations.
"""
from typing import List, Dict, Any

# Keywords that indicate medical advice/instructions
ADVICE_CUES: List[str] = [
    "खाएं",       # eat
    "न खाएं",     # don't eat
    "पीएं",       # drink
    "मत",         # don't
    "परहेज",      # avoid
    "आराम",       # rest
    "बचें",       # avoid
    "लेते रहें",  # keep taking
    "कम करें",    # reduce
    "ज्यादा न",   # not too much
    "सुबह",       # morning
    "शाम",        # evening
    "खाली पेट",   # empty stomach
    "खाने के बाद" # after eating
]


def extract_doctor_advice(entries: List[Dict[str, str]]) -> List[str]:
    """
    Extract medical advice from doctor's utterances.
    
    Strategy: If a doctor's sentence contains any advisory cue word,
    treat the entire sentence as advice. This captures both medication
    instructions and lifestyle recommendations.
    
    Args:
        entries: List of transcript entries with "speaker" and "text" keys
        
    Returns:
        List of advice strings (may contain duplicates if doctor repeats)
        
    Example:
        >>> entries = [
        ...     {"speaker": "doctor", "text": "दवा सुबह लें"},
        ...     {"speaker": "patient", "text": "ठीक है"}
        ... ]
        >>> extract_doctor_advice(entries)
        ['दवा सुबह लें']
    """
    advice: List[str] = []
    
    for entry in entries:
        # Only process doctor's speech
        if entry.get("speaker") != "doctor":
            continue
        
        text = entry.get("text", "").strip()
        if not text:
            continue
        
        # Check if any advisory cue appears in the sentence
        for cue in ADVICE_CUES:
            if cue in text:
                advice.append(text)
                break  # Don't add same sentence multiple times
    
    return advice

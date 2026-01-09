"""
Extracts medical advice from doctor's utterances.
Identifies advisory language patterns in Hindi medical consultations.
"""
from typing import List, Dict, Any

# Keywords that indicate medical advice/instructions
ADVICE_CUES = [
    # Medication / actions
    "लें", "ले", "लेनी", "लेते",
    "करें", "करे", "करना",

    # Restrictions / negatives
    "मत", "न", "ना",

    # Diet & fluids
    "खाएं", "पीएं", "पियो", "पिए", "पानी",

    # Rest / lifestyle
    "आराम", "परहेज", "बचें",

    # Timing / dosage
    "सुबह", "दोपहर", "शाम", "रात",
    "दिन में", "खाली पेट", "खाने के बाद",

    # Follow-up
    "फिर आ", "दिखा", "मिलें"
]

def extract_doctor_advice(transcript: List[Dict[str, str]]) -> List[str]:
    """Extract doctor's advice/instructions."""
    advice = []
    
    # First get diagnosis to avoid duplication
    from diagnosis_from_doctor_hi import extract_doctor_diagnosis
    diagnosis_lines = set(extract_doctor_diagnosis(transcript))
    
    for entry in transcript:
        if entry.get("speaker") != "doctor":
            continue
        
        text = entry.get("text", "").strip()
        
        # Skip if this was already captured as diagnosis
        if text in diagnosis_lines:
            continue
        
        # Rest of your existing advice extraction logic...
        if any(cue in text for cue in ADVICE_CUES):
            advice.append(text)
    
    return advice

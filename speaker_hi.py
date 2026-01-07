"""
Speaker detection using rule-based keyword matching.
Class-based to maintain conversation state per session.
"""
from typing import List

# Doctor-specific clinical language patterns
DOCTOR_CUES: List[str] = [
    "मैं आपकी जांच", "मैं आपकी जाँच", "जाँच कर रहा",
    "दवा", "लेनी है", "लें", "लिख रहा",
    "केस", "निदान", "डायग्नोसिस", "diagnosis", "लगता है"
]

# Doctor's imperative/advisory language
DOCTOR_IMPERATIVES: List[str] = [
    "खाएं", "न खाएं", "पीएं", "मत", "परहेज",
    "आराम", "बचें", "लेते रहें", "कम करें", "ज्यादा न"
]

# Patient-specific symptom/complaint language
PATIENT_CUES: List[str] = [
    "मुझे", "मेरे", "दर्द", "तकलीफ",
    "हो रहा", "हो रही", "है", "नहीं है", "नहीं हुई"
]


class SpeakerDetector:
    """
    Stateful speaker detection for a single consultation.
    
    Maintains conversation context: if the doctor was speaking,
    ambiguous sentences default to doctor (conversational continuity).
    
    This prevents spurious speaker switches when someone says just
    "हाँ" or "ठीक है" which could be either speaker.
    """
    
    def __init__(self):
        """Initialize with default assumption that patient speaks first."""
        self._last_speaker: str = "patient"
    
    def detect(self, text: str) -> str:
        """
        Detect speaker from utterance text, using conversation context.
        
        Priority order:
            1. Strong doctor signals (clinical terminology)
            2. Medical imperatives (advice/prescription language)
            3. Patient-specific language (complaints, symptoms)
            4. Ambiguous → use conversation context
        
        Args:
            text: Utterance text (Hindi)
            
        Returns:
            "doctor" or "patient"
            
        Example:
            >>> detector = SpeakerDetector()
            >>> detector.detect("मुझे सिर दर्द है")
            'patient'
            >>> detector.detect("दवा लें")
            'doctor'
        """
        text = text.strip()
        
        # Priority 1: Strong doctor signals (clinical language)
        if any(cue in text for cue in DOCTOR_CUES):
            self._last_speaker = "doctor"
            return "doctor"
        
        # Priority 2: Medical imperatives (advice/prescription language)
        if any(cue in text for cue in DOCTOR_IMPERATIVES):
            self._last_speaker = "doctor"
            return "doctor"
        
        # Priority 3: Patient-specific language (complaints, symptoms)
        if any(cue in text for cue in PATIENT_CUES):
            self._last_speaker = "patient"
            return "patient"
        
        # Priority 4: Ambiguous → use conversation context
        # (e.g., just "हाँ" or "ठीक है" continues current speaker)
        return self._last_speaker
    
    def reset(self):
        """Reset to default state for new consultation."""
        self._last_speaker = "patient"


# Legacy function for backward compatibility
# (in case any old code imports this)
_default_detector = SpeakerDetector()

def detect_speaker_hi(text: str) -> str:
    """
    Legacy function - use SpeakerDetector class for new code.
    
    Note: This uses a shared detector instance, so it won't work
    correctly for concurrent sessions. Use SpeakerDetector class instead.
    """
    return _default_detector.detect(text)

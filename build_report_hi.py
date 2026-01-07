"""
Orchestrates pipeline: raw transcript → normalized → segmented → structured JSON
"""
from typing import Dict, List, Any, Optional

from normalize_hi import normalize_hi
from segment_hi import segment_hi
from extract_hi import extract_symptoms, extract_medications


def build_json_hi(
    transcript: str,
    speaker: str = "patient",
    extra_text_for_meds: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert raw Hindi transcript into structured clinical data.
    
    Pipeline:
        1. Normalize: Remove filler words and clean text
        2. Segment: Split into individual sentences
        3. Extract: Pull out symptoms with negations
        4. Extract: Search for medications in broader context
    
    Args:
        transcript: Patient's raw speech text (Hindi)
        speaker: Speaker identifier (default: "patient")
        extra_text_for_meds: Include doctor's speech here for complete med history.
                            Useful because doctors often mention current medications.
    
    Returns:
        Dict with structure:
            {
                "speaker": str,
                "chief_complaint": str or None,
                "symptoms": List[Dict],  # [{name, location?, duration?}, ...]
                "negatives": List[str],  # Explicitly denied symptoms
                "medications": List[str]
            }
    """
    # Step 1: Clean filler words and normalize spacing
    normalized_text = normalize_hi(transcript)
    
    # Step 2: Split into sentences for entity extraction
    sentences = segment_hi(normalized_text)
    
    # Step 3: Extract symptoms and explicitly negated symptoms
    # Negations are important for differential diagnosis
    symptoms, negated_symptoms = extract_symptoms(sentences)
    
    # Step 4: Search broader text for medications
    # Use doctor's speech too since they mention current meds during consultation
    medication_search_text = extra_text_for_meds if extra_text_for_meds else normalized_text
    medications = extract_medications(medication_search_text)
    
    # Build structured output
    chief_complaint = symptoms[0]["name"] if symptoms else None
    
    return {
        "speaker": speaker,
        "chief_complaint": chief_complaint,
        "symptoms": symptoms,
        "negatives": negated_symptoms,
        "medications": medications
    }

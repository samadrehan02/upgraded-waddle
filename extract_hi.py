"""
Clinical entity extraction from segmented Hindi text.
Handles symptoms, locations, durations, medications, and negations.
"""
import re
from typing import List, Dict, Set, Tuple, Any

from medical_vocab_hi import (
    SYMPTOMS, NEGATIONS, LOCATIONS, 
    DURATION_PATTERNS, MEDICATIONS
)

# ==============================================================================
# PRECOMPILED PATTERNS (for performance)
# ==============================================================================
# Compile duration patterns once at module load instead of on every call
DURATION_PATTERNS_COMPILED = [re.compile(pattern) for pattern in DURATION_PATTERNS]


# ==============================================================================
# NEGATION DETECTION
# ==============================================================================
def has_negation(sentence: str, symptom_variants: List[str]) -> bool:
    """
    Check if sentence contains negated mention of symptom.
    
    Logic: If a negation word AND a symptom variant both appear
    in the same sentence, the symptom is negated.
    
    Args:
        sentence: Text sentence to check
        symptom_variants: List of possible names for the symptom
        
    Returns:
        True if symptom is explicitly negated
        
    Example:
        >>> has_negation("बुखार नहीं है", ["बुखार", "फीवर"])
        True
        >>> has_negation("बुखार है", ["बुखार", "फीवर"])
        False
    """
    has_negation_word = any(neg in sentence for neg in NEGATIONS)
    has_symptom = any(variant in sentence for variant in symptom_variants)
    
    return has_negation_word and has_symptom


# ==============================================================================
# SYMPTOM EXTRACTION
# ==============================================================================
def extract_symptoms(sentences: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Extract symptoms with context (location, duration) from sentence list.
    
    Strategy:
        - Scan each sentence for known symptom terms
        - If found and not negated, extract with location/duration from same sentence
        - Track negated symptoms separately (important for differential diagnosis)
    
    Args:
        sentences: List of segmented sentence strings
        
    Returns:
        Tuple of (positive_symptoms, negative_symptoms)
            positive_symptoms: List of dicts with 'name', 'location'?, 'duration'?
            negative_symptoms: List of explicitly negated symptom names
        
    Example:
        >>> sentences = ["छाती में दर्द है", "बाईं तरफ", "3 दिन से", "बुखार नहीं है"]
        >>> symptoms, negatives = extract_symptoms(sentences)
        >>> symptoms[0]
        {'name': 'छाती में दर्द', 'location': 'बाईं तरफ', 'duration': '3 दिन'}
        >>> negatives
        ['बुखार']
    """
    found_positive: Dict[str, Dict[str, str]] = {}  # canonical_name → symptom_dict
    found_negative: Set[str] = set()
    
    for sentence in sentences:
        # Check each known symptom against sentence
        for canonical, variants in SYMPTOMS.items():
            symptom_mentioned = any(variant in sentence for variant in variants) or canonical in sentence
            
            if not symptom_mentioned:
                continue
            
            # Negated mention (e.g., "बुखार नहीं है")
            if has_negation(sentence, variants):
                found_negative.add(canonical)
                continue
            
            # Positive mention - extract with context
            symptom_entry = found_positive.get(canonical, {"name": canonical})
            
            # Extract location if present in same sentence
            for location_name, location_keywords in LOCATIONS.items():
                if any(keyword in sentence for keyword in location_keywords):
                    symptom_entry["location"] = location_name
            
            # Extract duration if present (use precompiled patterns)
            for pattern in DURATION_PATTERNS_COMPILED:
                match = pattern.search(sentence)
                if match:
                    symptom_entry["duration"] = match.group(0)
                    break  # Only take first duration found
            
            found_positive[canonical] = symptom_entry
    
    symptoms = list(found_positive.values())
    
    # Only include negatives that weren't also mentioned positively
    # Handles cases like "पहले बुखार था, अब नहीं है" → fever is present, not absent
    negatives = [
        symptom for symptom in found_negative 
        if symptom not in found_positive
    ]
    
    return symptoms, negatives


# ==============================================================================
# MEDICATION EXTRACTION
# ==============================================================================
def extract_medications(text: str) -> List[str]:
    """
    Extract medications from text using simple substring matching.
    
    Strategy: Search entire transcript for known medication names.
    Works across both patient (mentions current meds) and doctor
    (prescribes new meds) speech.
    
    Args:
        text: Combined transcript text (typically includes both speakers)
        
    Returns:
        List of canonical medication names found
        
    Note: Current implementation has false positive risk (e.g., "पैन" 
    can appear in regular words). Future improvement: add word boundary
    checks or extract only from doctor utterances.
        
    Example:
        >>> extract_medications("मैं डोलो ले रहा हूं")
        ['पैरासिटामोल']
    """
    medications: List[str] = []
    
    for canonical, variants in MEDICATIONS.items():
        if any(variant in text for variant in variants):
            medications.append(canonical)
    
    return medications


# ==============================================================================
# DEPRECATED: CONTEXT ATTACHMENT
# ==============================================================================
def attach_context_and_confidence(
    symptoms: List[Dict[str, Any]], 
    sentences: List[str]
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Attach context from adjacent sentences to symptoms.
    
    This function handles cases where location/duration appear in separate
    sentences from the symptom mention. Currently unused in pipeline because
    the main extract_symptoms() function handles context well enough.
    
    Historical example:
        Sentence 1: "छाती में दर्द है"
        Sentence 2: "बाईं तरफ" ← This gets attached to symptom from S1
    
    Keeping this for reference in case we need cross-sentence context later.
    """
    last_symptom: Dict[str, Any] = None
    pending_context: Dict[str, str] = {}
    
    for sentence in sentences:
        # Check if current sentence mentions a symptom
        mentioned_symptom = None
        for symptom in symptoms:
            if symptom["name"] in sentence:
                mentioned_symptom = symptom
                break
        
        if mentioned_symptom:
            # Found symptom - boost confidence and attach pending context
            last_symptom = mentioned_symptom
            last_symptom["confidence"] = max(
                last_symptom.get("confidence", 0.9), 0.9
            )
            
            # Apply any pending context from previous sentences
            for key, value in pending_context.items():
                last_symptom[key] = value
                last_symptom["confidence"] = min(
                    last_symptom["confidence"] + 0.05, 1.0
                )
            
            pending_context.clear()
            continue
        
        # No symptom in current sentence
        if not last_symptom:
            # No context yet - buffer location/duration for next symptom
            for location_name, location_keywords in LOCATIONS.items():
                if any(keyword in sentence for keyword in location_keywords):
                    pending_context["location"] = location_name
            
            for pattern in DURATION_PATTERNS_COMPILED:
                match = pattern.search(sentence)
                if match:
                    pending_context["duration"] = match.group(0)
            continue
        
        # Have context from previous symptom - try to attach additional details
        for location_name, location_keywords in LOCATIONS.items():
            if any(keyword in sentence for keyword in location_keywords):
                if "location" not in last_symptom:
                    last_symptom["location"] = location_name
                    last_symptom["confidence"] = min(
                        last_symptom["confidence"] + 0.05, 1.0
                    )
        
        for pattern in DURATION_PATTERNS_COMPILED:
            match = pattern.search(sentence)
            if match and "duration" not in last_symptom:
                last_symptom["duration"] = match.group(0)
                last_symptom["confidence"] = min(
                    last_symptom["confidence"] + 0.05, 1.0
                )
    
    # Set default confidence for all symptoms
    for symptom in symptoms:
        symptom.setdefault("confidence", 0.85)
    
    return symptoms

"""
Clinical entity extraction from segmented Hindi text.
Handles symptoms, locations, durations, medications, and negations.
"""

import re
from typing import List, Dict, Set, Tuple, Any
from typing import Optional

from medical_vocab_hi import (
    SYMPTOMS,
    NEGATIONS,
    LOCATIONS,
    DURATION_PATTERNS,
    MEDICATIONS,
)
# Hindi spoken numbers commonly used for age
HINDI_NUMBER_WORDS = {
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "छह": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "तेईस": 23,
    "चौबीस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "अट्ठाईस": 28,
    "उनतीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "चौंतीस": 34,
    "पैंतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "अड़तीस": 38,
    "उनतालीस": 39,
    "चालीस": 40,
    "पैंतालीस": 45,
    "पचास": 50,
    "पचपन": 55,
    "साठ": 60,
    "पैंसठ": 65,
    "सत्तर": 70,
    "पचहत्तर": 75,
    "अस्सी": 80,
    "पचासी": 85,
    "नब्बे": 90
}

DURATION_PATTERNS_COMPILED = [re.compile(p) for p in DURATION_PATTERNS]

# ------------------------------------------------------------------------------
# Patient demographics extraction (ADD-ON, non-clinical)
# ------------------------------------------------------------------------------

EN_NAME_PATTERN = re.compile(
    r"patient\s+name\s*(?:is\s*)?([A-Za-z]+)",
    re.IGNORECASE
)

EN_AGE_PATTERN = re.compile(
    r"patient\s+age\s*(?:is\s*)?(\d{1,3})",
    re.IGNORECASE
)

HI_NAME_PATTERNS = [
    r"patient\s+ka\s+naam\s+([^\s]+)",
    r"मरीज\s+का\s+नाम\s+([^\s]+)",
    r"मेरा\s+नाम\s+([^\s]+)",
    r"नाम\s+([^\s]+)\s+है",
    r"पेशेंट\s+नेम\s+([^\s]+)"
]


HI_SELF_NAME_PATTERN = re.compile(
    r"मैं\s+([^\s]+)\s+(?:हूँ|हूं)",
    re.UNICODE
)

AGE_PATTERNS = [
    r"patient\s+age\s*(?:is\s*)?(\d{1,3})",
    r"मेरी\s+उम्र\s*(\d{1,3})",
    r"उम्र\s*(\d{1,3})",
    r"(\d{1,3})\s*साल"
]
# Negation detection (pre + post symptom, tight window)

def has_negation(sentence: str, symptom_variants: List[str]) -> bool:
    """
    Detect explicit negation of a symptom within a tight local window
    BEFORE or AFTER the symptom phrase.
    """

    for variant in symptom_variants:
        if variant not in sentence:
            continue

        idx = sentence.find(variant)

        before = sentence[max(0, idx - 15):idx]
        after = sentence[idx + len(variant): idx + len(variant) + 15]

        if any(neg in before for neg in NEGATIONS):
            return True

        if any(neg in after for neg in NEGATIONS):
            return True

    return False


# Symptom extraction with duration attachment
def extract_symptoms(sentences: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Extract symptoms with location and duration.
    Handles:
    - Explicit negation (pre/post)
    - Standalone duration sentences
    """

    found_positive: Dict[str, Dict[str, str]] = {}
    found_negative: Set[str] = set()

    pending_duration: str | None = None

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Capture standalone duration (e.g. "तीन दिन से")
        for pattern in DURATION_PATTERNS_COMPILED:
            match = pattern.search(sentence)
            if match:
                # Only buffer if this sentence has NO symptom mention
                if not any(
                    variant in sentence
                    for variants in SYMPTOMS.values()
                    for variant in variants
                ):
                    pending_duration = match.group(0)
                    break

        # Symptom detection
        for canonical, variants in SYMPTOMS.items():
            mentioned = any(v in sentence for v in variants)

            if not mentioned:
                continue

            # Explicit negation → negative symptom
            if has_negation(sentence, variants):
                found_negative.add(canonical)
                continue

            # Positive symptom
            symptom_entry = found_positive.get(canonical, {"name": canonical})

            # Attach pending duration
            if pending_duration and "duration" not in symptom_entry:
                symptom_entry["duration"] = pending_duration
                pending_duration = None

            # Extract duration from same sentence
            for pattern in DURATION_PATTERNS_COMPILED:
                match = pattern.search(sentence)
                if match:
                    symptom_entry["duration"] = match.group(0)
                    break

            # Extract location
            for loc_name, loc_variants in LOCATIONS.items():
                if any(v in sentence for v in loc_variants):
                    symptom_entry["location"] = loc_name

            found_positive[canonical] = symptom_entry

    # Do not list negatives that also appear positively
    negatives = [
        s for s in found_negative
        if s not in found_positive
    ]

    return list(found_positive.values()), negatives


# Medication extraction (simple substring matching)
def extract_medications(text: str) -> List[str]:
    """
    Extract medications using canonical vocab matching.
    """

    medications: List[str] = []

    for canonical, variants in MEDICATIONS.items():
        if any(v in text for v in variants):
            medications.append(canonical)

    return medications

def extract_patient_name(text: str) -> Optional[str]:
    """
    Extract patient name from full transcript text.

    Priority:
    1. English explicit command: "patient name Samarth"
    2. Hindi explicit phrases: "मेरा नाम समार्थ है"
    3. Hindi self-intro: "मैं समार्थ हूँ"
    """

    if not text:
        return None

    # 1. English explicit
    m = EN_NAME_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if len(name) >= 2:
            return name

    # 2. Hindi explicit
    for pat in HI_NAME_PATTERNS:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            if len(name) >= 2:
                return name

    # 3. Hindi self introduction
    m = HI_SELF_NAME_PATTERN.search(text)
    if m:
        name = m.group(1).strip()
        if len(name) >= 2:
            return name

    return None

def extract_patient_age(text: str) -> Optional[int]:
    """
    Extract patient age from full transcript text.
    Supports numeric and spoken Hindi numbers.
    """

    if not text:
        return None

    for pat in AGE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                age = int(m.group(1))
                if 0 < age < 120:
                    return age
            except ValueError:
                pass

    for word, value in HINDI_NUMBER_WORDS.items():
        if f"{word} साल" in text or f"{word} वर्ष" in text:
            if 0 < value < 120:
                return value

    return None

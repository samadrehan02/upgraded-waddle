"""
Text normalization for Hindi medical transcripts.
Removes non-clinical filler words while preserving medical content.
"""
from typing import List

# Filler phrases and words to remove
# Order matters: check longer phrases first
DROP_LINES = [
    "आवाज आ रही है",  # Longer phrases first
    "आवाज़ आ रही है",
    "हेलो",
    "हैलो",
    "hello",
    "जी",
    "हाँ",
    "हां",
    "ठीक है",
    "समझ गया",
    "समझ गयी",
    "प्रोफाइल",
    "test",
    "testing"
]


def is_filler_line(line: str) -> bool:
    """
    Check if line is pure filler (should be removed).
    
    Strategy:
        1. Check if entire line matches a filler phrase exactly
        2. Check if line is composed entirely of filler words
    
    Returns True ONLY if line contains ONLY filler content.
    Important: Don't remove lines with medical content + filler.
    
    Args:
        line: Single line of text (already lowercased)
        
    Returns:
        True if line should be removed, False otherwise
    """
    line_stripped = line.strip()
    
    if not line_stripped:
        return True
    
    # Check if entire line is a known filler phrase
    for filler in DROP_LINES:
        if line_stripped == filler:
            return True
        # Also check if line is just repetitions of a filler
        # e.g., "हेलो हेलो" or "हेलो हेलो हेलो"
        if line_stripped.replace(filler, "").strip() == "":
            return True
    
    # Check if line contains ONLY filler phrases (multiple phrases combined)
    # e.g., "हेलो आवाज आ रही है"
    remaining = line_stripped
    for filler in DROP_LINES:
        remaining = remaining.replace(filler, "")
    
    # If after removing all fillers, nothing meaningful remains, it's a filler line
    remaining_cleaned = remaining.strip()
    if not remaining_cleaned or remaining_cleaned in [",", ".", "!", "?", " "]:
        return True
    
    return False


def normalize_hi(text: str) -> str:
    """
    Clean Hindi transcript by removing non-clinical utterances.
    
    Preserves all medical content (symptoms, negations, locations).
    Only removes pure filler lines like "hello", "can you hear me".
    
    Args:
        text: Raw Hindi transcript text
        
    Returns:
        Cleaned text with fillers removed, lowercased
        
    Example:
        >>> normalize_hi("हेलो\\nमुझे बुखार है\\nआवाज आ रही है")
        'मुझे बुखार है'
    """
    lines = text.split("\n")
    cleaned: List[str] = []
    
    for line in lines:
        normalized_line = line.strip().lower()
        
        if not normalized_line:
            continue
        
        # Skip pure filler lines
        if is_filler_line(normalized_line):
            continue
        
        cleaned.append(normalized_line)
    
    return "\n".join(cleaned)

"""
Text segmentation for Hindi medical transcripts.
Splits on natural sentence boundaries (commas, newlines).
"""
import re
from typing import List


def segment_hi(text: str) -> List[str]:
    """
    Split Hindi text into sentences using common delimiters.
    
    Strategy: Split on commas and newlines, which work well for
    spoken medical Hindi where sentences are often short statements.
    
    Args:
        text: Normalized Hindi text
        
    Returns:
        List of non-empty sentence strings
        
    Example:
        >>> segment_hi("सिर दर्द है, 3 दिन से\\nबुखार नहीं है")
        ['सिर दर्द है', '3 दिन से', 'बुखार नहीं है']
    """
    parts = re.split(r"[,\n]", text)
    return [p.strip() for p in parts if p.strip()]

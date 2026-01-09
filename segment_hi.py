"""
Text segmentation for Hindi medical transcripts.
Splits on natural sentence boundaries (commas, newlines).
"""
import re
from typing import List

def segment_hi(text: str) -> List[str]:
    parts = re.split(r"[,\n]", text)
    return [p.strip() for p in parts if p.strip()]

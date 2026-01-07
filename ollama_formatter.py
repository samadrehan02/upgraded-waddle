"""
Formats structured JSON into professional OPD notes using Ollama LLM.
Strict prompt engineering prevents hallucination of clinical facts.
"""
import json
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODEL_NAME = "llama3.1:8b" 

OLLAMA_TIMEOUT_SECONDS = 30
MAX_JSON_SIZE_BYTES = 50000

# ENHANCED SYSTEM PROMPT - More explicit examples
SYSTEM_PROMPT = """You are a medical documentation assistant. Your ONLY job is to convert the Hindi medical data below into a professional English OPD note.

⚠️ CRITICAL RULES - VIOLATION CAUSES MEDICAL HARM:

1. USE ONLY THE JSON DATA BELOW
   - Every sentence you write MUST come from the JSON
   - If JSON says {"symptoms": [{"name": "बुखार"}]}, you can ONLY write "Fever"
   - If JSON has no age → DO NOT write age
   - If JSON has no gender → DO NOT write gender
   - If JSON has no duration → DO NOT write duration

2. FORBIDDEN ACTIONS:
   ❌ DO NOT invent patient demographics
   ❌ DO NOT invent symptom details not in JSON
   ❌ DO NOT invent differential diagnoses
   ❌ DO NOT invent medication instructions not in advice
   ❌ DO NOT invent follow-up plans
   ❌ DO NOT add medical reasoning or explanations

3. TRANSLATION RULES:
   - Translate Hindi terms to English medical terminology
   - बुखार → Fever (not "viral fever" or "high fever" unless specified)
   - कमजोरी → Weakness (not "fatigue" or "generalized weakness")
   - परसों से → since 2 days ago (exact translation)

4. FORMAT:
   Use these sections ONLY if data exists:
   - Chief Complaint: [translate from chief_complaint]
   - Symptoms: [list from symptoms array, include duration/location if present]
   - Negative Findings: [list from negatives array]
   - Current Medications: [list from medications array]
   - Assessment: [list from diagnosis array - translate but don't expand]
   - Plan: [list from advice array - translate but don't add details]

5. EXAMPLES OF CORRECT BEHAVIOR:

INPUT JSON:
{
  "chief_complaint": "बुखार",
  "symptoms": [{"name": "बुखार", "duration": "परसों से"}],
  "medications": ["पैरासिटामोल"],
  "diagnosis": ["बुखार का केस"],
  "advice": ["आराम करें"]
}

CORRECT OUTPUT:
Chief Complaint: Fever

Symptoms:
- Fever (since 2 days ago)

Current Medications:
- Paracetamol

Assessment:
- Fever case

Plan:
- Rest

❌ WRONG OUTPUT (DO NOT DO THIS):
Chief Complaint: High fever with chills

History: The patient is a 30-year-old male presenting with fever for the past 3 days...
[This is WRONG because: invented "high", "chills", "30-year-old male", "3 days"]

REMEMBER: If you add ANY information not in the JSON, you are violating medical documentation standards.
"""


def validate_input(structured_json: Dict[str, Any]) -> None:
    """Validate input structure and size."""
    if not isinstance(structured_json, dict):
        raise ValueError("Input must be a dictionary")
    
    json_str = json.dumps(structured_json, ensure_ascii=False, indent=2)
    json_size = len(json_str.encode('utf-8'))
    
    if json_size > MAX_JSON_SIZE_BYTES:
        logger.warning(f"Input JSON size {json_size} exceeds limit {MAX_JSON_SIZE_BYTES}")
        raise ValueError(f"Input JSON too large (>{MAX_JSON_SIZE_BYTES/1000}KB)")


def call_ollama(prompt: str) -> str:
    """Execute Ollama CLI with timeout."""
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error(f"Ollama returned non-zero exit code: {error_msg}")
            raise RuntimeError(f"Ollama execution failed: {error_msg}")
        
        return result.stdout.strip()
    
    except subprocess.TimeoutExpired:
        logger.error(f"Ollama timeout after {OLLAMA_TIMEOUT_SECONDS}s")
        raise RuntimeError(f"Report generation timed out")
    
    except FileNotFoundError:
        logger.error("Ollama CLI not found")
        raise RuntimeError("Ollama not installed or not in PATH")


def validate_output(output: str, input_json: Dict[str, Any]) -> str:
    """
    Validate and clean LLM output.
    
    Returns cleaned output with warning if hallucination suspected.
    """
    if not output:
        raise RuntimeError("Ollama returned empty output")
    
    if "HALLUCINATION DETECTED" in output or "INVALID OUTPUT" in output:
        logger.error("LLM explicitly flagged violation")
        raise RuntimeError("LLM violated documentation rules")
    
    # Detect hallucination patterns
    input_str = json.dumps(input_json, ensure_ascii=False).lower()
    output_lower = output.lower()
    
    hallucinations_found = []
    
    # Check for invented demographics
    demographic_terms = ["year old", "years old", "male", "female", "aged"]
    for term in demographic_terms:
        if term in output_lower and term not in input_str:
            hallucinations_found.append(f"Invented demographic: '{term}'")
    
    # Check for invented diagnoses
    diagnosis_terms = ["viral", "bacterial", "infection", "influenza", "respiratory"]
    has_diagnosis = input_json.get("diagnosis", [])
    if has_diagnosis:
        diagnosis_str = " ".join(has_diagnosis).lower()
        for term in diagnosis_terms:
            if term in output_lower and term not in diagnosis_str and term not in input_str:
                hallucinations_found.append(f"Invented diagnosis detail: '{term}'")
    
    # Check for invented advice
    advice_terms = ["hydrat", "follow-up", "monitor", "return", "call", "if worsens"]
    has_advice = input_json.get("advice", [])
    if has_advice:
        advice_str = " ".join(has_advice).lower()
        for term in advice_terms:
            if term in output_lower and term not in advice_str:
                hallucinations_found.append(f"Invented advice: '{term}'")
    
    if hallucinations_found:
        warning = "\n\n⚠️ WARNING: Possible hallucinations detected:\n"
        warning += "\n".join(f"  - {h}" for h in hallucinations_found)
        warning += "\n\nPlease verify all information against the original transcript."
        logger.warning(f"Hallucinations detected: {hallucinations_found}")
        return output + warning
    
    return output


def generate_opd_note(structured_json: Dict[str, Any]) -> str:
    """
    Generate formatted OPD note from structured clinical data.
    
    Args:
        structured_json: Dict with symptoms, medications, diagnosis, advice
        
    Returns:
        Formatted OPD note text
        
    Raises:
        ValueError: If input invalid
        RuntimeError: If Ollama fails or LLM hallucinates
    """
    validate_input(structured_json)
    
    # Build concise prompt
    json_str = json.dumps(structured_json, ensure_ascii=False, indent=2)
    prompt = f"""{SYSTEM_PROMPT}

INPUT JSON (your only source of information):
{json_str}

Generate a concise OPD note. Use ONLY information from the JSON above. Do not add any details.
"""
    
    try:
        output = call_ollama(prompt)
        output = validate_output(output, structured_json)
        return output
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise RuntimeError(f"Report generation failed: {str(e)}")

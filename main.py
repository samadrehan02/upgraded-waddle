"""
FastAPI server for real-time Hindi medical transcription via WebSocket.
Each connection gets isolated session to prevent data mixing between consultations.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import logging
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from vosk import Model, KaldiRecognizer

from build_report_hi import build_json_hi
from speaker_hi import SpeakerDetector
from diagnosis_from_doctor_hi import extract_doctor_diagnosis
from advice_from_doctor_hi import extract_doctor_advice
from ollama_formatter import generate_opd_note

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "vosk-model-hi-0.22"  # TODO: Move to environment variables
SAMPLE_RATE = 48000  # Must match frontend audio settings
TEMPLATE_PATH = "templates/index.html"
STATIC_DIR = "static"

# Initialize FastAPI
app = FastAPI(title="Hindi Medical Transcription API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load Vosk model once at startup (expensive ~2-3s)
logger.info(f"Loading Vosk model from {MODEL_PATH}")
model = Model(MODEL_PATH)

# Session storage: prevents concurrent consultations from mixing data
# Structure: {session_id: {"transcript": [...], "speaker_detector": SpeakerDetector()}}
sessions: Dict[str, Dict[str, Any]] = {}


def create_recognizer() -> KaldiRecognizer:
    """Create a new Vosk recognizer instance with standard settings."""
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    recognizer.SetPartialWords(True)
    return recognizer


def create_session(session_id: str) -> None:
    """Initialize a new consultation session."""
    sessions[session_id] = {
        "transcript": [],
        "speaker_detector": SpeakerDetector()
    }
    logger.info(f"Created session: {session_id}")


def cleanup_session(session_id: str) -> None:
    """Clean up session data to prevent memory leaks."""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Cleaned up session: {session_id}")


def reset_session(session_id: str, recognizer: KaldiRecognizer) -> KaldiRecognizer:
    """Reset session state for new consultation while keeping session alive."""
    sessions[session_id]["transcript"].clear()
    sessions[session_id]["speaker_detector"].reset()
    return create_recognizer()


def extract_speaker_lines(transcript: list, speaker: str) -> list:
    """Extract all text from a specific speaker."""
    return [entry["text"] for entry in transcript if entry["speaker"] == speaker]


def build_structured_data(session_id: str) -> Dict[str, Any]:
    """
    Process full consultation transcript into structured clinical data.
    
    Returns dict with symptoms, medications, diagnosis, and advice.
    """
    session_data = sessions[session_id]
    full_transcript = session_data["transcript"]
    
    # Separate patient speech for symptom extraction (avoids polluting with doctor's advice)
    patient_lines = extract_speaker_lines(full_transcript, "patient")
    combined_patient = "\n".join(patient_lines)
    
    # Keep all speech for medication extraction (doctor mentions current meds)
    all_lines = [entry["text"] for entry in full_transcript]
    combined_all = "\n".join(all_lines)
    
    # Extract clinical entities
    structured = build_json_hi(combined_patient, extra_text_for_meds=combined_all)
    
    # Extract doctor-specific information
    diagnoses = extract_doctor_diagnosis(full_transcript)
    advice = extract_doctor_advice(full_transcript)
    
    # Deduplicate while preserving order
    structured["diagnosis"] = list(dict.fromkeys(diagnoses))
    structured["advice"] = list(dict.fromkeys(advice))
    
    return structured


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve main UI."""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Real-time audio → text transcription endpoint.
    
    Protocol:
      Client sends: audio chunks (binary) or "stop" (text)
      Server sends: {"type": "partial"|"transcript"|"structured"|"error", ...}
    """
    await ws.accept()
    
    # Create unique session for this consultation
    session_id = str(uuid4())
    create_session(session_id)
    recognizer = create_recognizer()
    
    try:
        while True:
            message = await ws.receive()
            
            # Handle stop command: process full consultation
            if "text" in message and message["text"] == "stop":
                structured = build_structured_data(session_id)
                
                await ws.send_json({
                    "type": "structured",
                    "data": structured
                })
                
                # Reset for next consultation in same session
                recognizer = reset_session(session_id, recognizer)
                continue
            
            # Handle audio chunk: transcribe and detect speaker
            if "bytes" in message:
                data = message["bytes"]
                
                if recognizer.AcceptWaveform(data):
                    # Complete utterance detected
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        speaker = sessions[session_id]["speaker_detector"].detect(text)
                        sessions[session_id]["transcript"].append({
                            "speaker": speaker,
                            "text": text
                        })
                        
                        await ws.send_json({
                            "type": "transcript",
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "text": text,
                            "speaker": speaker
                        })
                else:
                    # Partial result (user still speaking)
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    
                    if partial_text:
                        await ws.send_json({
                            "type": "partial",
                            "text": partial_text
                        })
    
    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected by client")
        cleanup_session(session_id)
    
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        try:
            await ws.send_json({
                "type": "error",
                "message": "Processing error occurred"
            })
        except:
            pass  # Connection may already be closed
        cleanup_session(session_id)


@app.post("/generate-report")
def generate_report(structured: dict = Body(...)):
    """Generate formatted OPD note from structured JSON using Ollama."""
    try:
        note = generate_opd_note(structured)
        return {"note": note}
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return {"error": "Failed to generate report"}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""FastAPI server for real-time Hindi medical transcription via WebSocket. Stable, low-latency streaming ASR with Vosk."""
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
from fastapi import HTTPException

MODEL_PATH = "vosk-model-hi-0.22"
SAMPLE_RATE = 16000
TEMPLATE_PATH = "templates/index.html"
STATIC_DIR = "static"
REQUIRED_STRUCTURED_KEYS = {
    "symptoms",
    "negatives",
    "medications",
    "diagnosis",
    "advice",
}
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = Model(MODEL_PATH)

sessions: Dict[str, Dict[str, Any]] = {}

def new_recognizer():
    """Create single recognizer with word timestamps enabled"""
    r = KaldiRecognizer(model, SAMPLE_RATE)
    r.SetWords(True)  # Enable word-level timestamps
    r.SetPartialWords(False)  # Keep partials clean
    return r

def create_session(sid: str):
    sessions[sid] = {
        "transcript": [],
        "speaker_detector": SpeakerDetector(),
        "audio": bytearray()
    }

def cleanup_session(sid: str):
    sessions.pop(sid, None)

def validate_structured_payload(structured: dict):
    if not isinstance(structured, dict):
        raise HTTPException(
            status_code=400,
            detail="Structured payload must be a JSON object"
        )

    missing = REQUIRED_STRUCTURED_KEYS - structured.keys()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required structured keys: {sorted(missing)}"
        )

    # Type safety (non-breaking, strict)
    for key in REQUIRED_STRUCTURED_KEYS:
        if not isinstance(structured.get(key), list):
            raise HTTPException(
                status_code=400,
                detail=f"'{key}' must be a list"
            )

@app.get("/", response_class=HTMLResponse)
def index():
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    sid = str(uuid4())
    create_session(sid)
    recognizer = new_recognizer()

    try:
        while True:
            msg = await ws.receive()
            sess = sessions[sid]

            if "text" in msg and msg["text"] == "stop":
                patient = "\n".join(t["text"] for t in sess["transcript"] if t["speaker"] == "patient")
                all_text = "\n".join(t["text"] for t in sess["transcript"])
                structured = build_json_hi(patient, extra_text_for_meds=all_text)
                structured["diagnosis"] = extract_doctor_diagnosis(sess["transcript"])
                structured["advice"] = extract_doctor_advice(sess["transcript"])

                await ws.send_json({"type": "structured", "data": structured})

                sess["transcript"].clear()

                recognizer = new_recognizer()
                sess["audio"].clear()
                sess["speaker_detector"].reset()
                continue

            if "bytes" in msg:
                data = msg["bytes"]
                sess["audio"].extend(data)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        words = result.get("result", [])
                        speaker = sess["speaker_detector"].detect(text)

                        entry = {
                            "speaker": speaker,
                            "text": text,
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "words": words
                        }

                        sess["transcript"].append(entry)

                        await ws.send_json({
                            "type": "transcript",
                            "time": entry["time"],
                            "speaker": speaker,
                            "text": text
                        })

                    sess["audio"].clear()

                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()

                    if partial_text:
                        await ws.send_json({
                            "type": "partial",
                            "text": partial_text
                        })

    except WebSocketDisconnect:
        cleanup_session(sid)
    except Exception:
        logger.exception("WebSocket error")
        cleanup_session(sid)

@app.post("/generate-report")
def generate_report(structured: dict = Body(...)):
    validate_structured_payload(structured)
    return {"note": generate_opd_note(structured)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

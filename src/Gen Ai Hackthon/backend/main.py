import os
import uuid
import subprocess
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from gtts import gTTS   # ✅ Added gTTS for multilingual TTS
from dotenv import load_dotenv

from utils import (
    AUDIO_DIR,
    DATA_DIR,
    extract_text_from_pdf,
    extract_text_from_image,
    save_upload,
    save_metadata,
    list_history,
)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
app = FastAPI()

# ----------------------------
# CORS (frontend -> backend)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change "*" -> your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("echoverse")

# ----------------------------
# Config
# ----------------------------
HF_API_KEY = os.environ.get("HF_API_KEY")
HF_MODEL = os.environ.get("HF_MODEL", "ibm-granite/granite-3.3-2b-instruct")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

AVAILABLE_MODELS = {
    "granite": {"type": "hf", "name": HF_MODEL},
    "gemini": {"type": "gemini", "name": "gemini-2.0-flash"},
}

# ✅ Supported languages for TTS
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Telugu": "te",
    "Kannada": "kn",
    "Hindi": "hi",
}

# ----------------------------
# Pydantic model
# ----------------------------
class GenerateRequest(BaseModel):
    text: str
    title: str | None = None
    llm_system_prompt: str | None = None
    model_choice: str = "granite"
    language: str = "en"

# ----------------------------
# Hugging Face Granite
# ----------------------------
def call_llm_hf(narration_text: str, system_prompt: str | None = None, model_name: str = HF_MODEL):
    if not HF_API_KEY:
        msg = "LLM not configured (HF_API_KEY missing)."
        logger.warning(msg)
        return narration_text, msg

    payload_text = narration_text
    if system_prompt:
        payload_text = f"{system_prompt}\n\n{narration_text}"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    endpoint = f"https://api-inference.huggingface.co/models/{model_name}"
    try:
        r = requests.post(endpoint, headers=headers, json={"inputs": payload_text}, timeout=300)
        r.raise_for_status()
        data = r.json()

        output = ""
        if isinstance(data, list) and len(data) > 0:
            output = data[0].get("generated_text", "")

        logger.info("✅ Granite LLM (HF) call succeeded")
        return output or narration_text, "success"
    except Exception as e:
        logger.exception("❌ Granite LLM (HF) call failed")
        return narration_text, f"failed: {e}"

# ----------------------------
# Google Gemini API
# ----------------------------
def call_llm_gemini(narration_text: str, system_prompt: str | None = None, model_name: str = "gemini-2.0-flash"):
    if not GEMINI_API_KEY:
        msg = "LLM not configured (GEMINI_API_KEY missing)."
        logger.warning(msg)
        return narration_text, msg

    payload_text = narration_text
    if system_prompt:
        payload_text = f"{system_prompt}\n\n{narration_text}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"

    try:
        r = requests.post(endpoint, headers=headers, json={"contents": [{"parts": [{"text": payload_text}]}]}, timeout=60)
        r.raise_for_status()
        data = r.json()

        output = ""
        if "candidates" in data and len(data["candidates"]) > 0:
            output = data["candidates"][0]["content"]["parts"][0]["text"]

        logger.info("✅ Gemini LLM call succeeded")
        return output or narration_text, "success"
    except Exception as e:
        logger.exception("❌ Gemini LLM call failed")
        return narration_text, f"failed: {e}"

# ----------------------------
# Model Dispatcher
# ----------------------------
def call_llm(model_choice: str, narration_text: str, system_prompt: str | None = None):
    model_info = AVAILABLE_MODELS.get(model_choice)
    if not model_info:
        return narration_text, f"Model {model_choice} not available"

    if model_info["type"] == "hf":
        return call_llm_hf(narration_text, system_prompt, model_info["name"])
    elif model_info["type"] == "gemini":
        return call_llm_gemini(narration_text, system_prompt, model_info["name"])
    else:
        return narration_text, "Unsupported model type"

# ----------------------------
# TTS (gTTS multilingual)
# ----------------------------
def synthesize_tts(text: str, out_path_mp3: str, language: str = "en"):
    try:
        tts = gTTS(text=text, lang=language)
        tts.save(out_path_mp3)
        logger.info(f"TTS written to {out_path_mp3} (lang={language})")
    except Exception:
        logger.exception("TTS synthesis failed")
        raise

# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "supported_languages": SUPPORTED_LANGUAGES,
    }

# ----------------------------
# Generate Audiobook
# ----------------------------
@app.post("/generate")
async def generate_audio(
    request: Request,
    title: str = Form(None),
    llm_system_prompt: str = Form(None),
    model_choice: str = Form("granite"),
    language: str = Form("en"),  # ✅ language selection
    file: UploadFile | None = File(None),
    raw_text: str | None = Form(None),
):
    try:
        source = "raw_text"
        text = (raw_text or "").strip()

        # Handle uploaded file
        if file:
            filename = (file.filename or "upload").lower()
            saved = await save_upload(file, DATA_DIR, suggested_name=filename)
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(saved)
                source = "pdf"
            elif filename.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                try:
                    text = extract_text_from_image(saved)
                    source = "image"
                except Exception as e:
                    logger.exception("OCR failed")
                    return JSONResponse({"error": f"OCR failed: {e}"}, status_code=500)
            else:
                try:
                    with open(saved, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read()
                    source = "file"
                except Exception as e:
                    logger.exception("Reading uploaded file failed")
                    return JSONResponse({"error": f"Failed to read uploaded file: {e}"}, status_code=500)

        if not text:
            return JSONResponse({"error": "No input text detected"}, status_code=400)

        # Call selected LLM
        loop = asyncio.get_running_loop()
        final_text, llm_status = await loop.run_in_executor(None, call_llm, model_choice, text, llm_system_prompt)

        # TTS synthesis (direct to mp3)
        uid = uuid.uuid4().hex
        out_mp3 = AUDIO_DIR / f"{uid}.mp3"

        await loop.run_in_executor(None, synthesize_tts, final_text, str(out_mp3), language)

        # Save metadata
        record = {
            "id": uid,
            "title": title or f"Audiobook {uid}",
            "filename": str(Path(out_mp3).name),
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": {
                "length_chars": len(final_text),
                "llm_used": True,
                "llm_status": llm_status,
                "model_choice": model_choice,
                "language": language,
            },
        }
        save_metadata(record)
        logger.info(f"Saved metadata for {uid} (file={record['filename']}, lang={language}, llm=True, status={llm_status}, model={model_choice})")

        return {
            "id": uid,
            "filename": record["filename"],
            "download_url": f"/download/{uid}",
            "llm_used": True,
            "llm_status": llm_status,
            "model_choice": model_choice,
            "language": language,
        }

    except Exception as e:
        logger.exception("Unexpected error during /generate")
        return JSONResponse({"error": f"Internal server error: {e}"}, status_code=500)

# ----------------------------
# Download Endpoint
# ----------------------------
@app.get("/download/{uid}")
def download(uid: str):
    rows = list_history()
    match = next((r for r in rows if r['id'] == uid), None)
    if not match:
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = AUDIO_DIR / match['filename']
    if not path.exists():
        return JSONResponse({"error": "File not found on server"}, status_code=404)
    ext = path.suffix.lower()
    media_type = "audio/mpeg" if ext in (".mp3",) else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=path.name)

# ----------------------------
# History Endpoint
# ----------------------------
@app.get("/history")
def history():
    rows = list_history()
    logger.info(f"/history requested — returning {len(rows)} records")
    return rows

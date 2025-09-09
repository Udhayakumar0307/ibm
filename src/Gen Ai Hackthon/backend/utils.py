import os
import sqlite3
import uuid
import json
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from fastapi import UploadFile

# ----------------------------
# Directories
# ----------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = DATA_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "metadata.db"

# ----------------------------
# Database Utilities
# ----------------------------
def init_db():
    """Initialize SQLite database with audiobooks table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS audiobooks (
            id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            source TEXT,
            created_at TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def save_metadata(record: dict):
    """Save a new audiobook metadata record into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO audiobooks (id, title, filename, source, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
        (
            record["id"],
            record.get("title"),
            record.get("filename"),
            record.get("source"),
            record.get("created_at"),
            json.dumps(record.get("metadata", {})),
        ),
    )
    conn.commit()
    conn.close()


def list_history():
    """Return all audiobook metadata records (newest first)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, title, filename, source, created_at, metadata FROM audiobooks ORDER BY created_at DESC"
    )
    rows = c.fetchall()
    conn.close()

    out = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "title": r[1],
                "filename": r[2],
                "source": r[3],
                "created_at": r[4],
                "metadata": json.loads(r[5] or "{}"),
            }
        )
    return out

# ----------------------------
# File/Text Utilities
# ----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    texts = [page.get_text() for page in doc]
    return "\n\n".join(texts)


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Automatically sets path for Windows if needed.
    """
    if os.name == "nt":  # Windows
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in possible_paths:
            if Path(p).exists():
                pytesseract.pytesseract.tesseract_cmd = p
                break
    else:
        # On Linux/Mac, assume tesseract is in PATH
        pytesseract.pytesseract.tesseract_cmd = "tesseract"

    # Final check: is it callable?
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        raise RuntimeError(
            f"OCR failed: Ensure Tesseract is installed and in PATH. Details: {e}"
        )


async def save_upload(file_obj: UploadFile, dest_dir: Path, suggested_name: Optional[str] = None) -> str:
    """
    Save uploaded file to disk (async-compatible for FastAPI UploadFile).
    Returns the full file path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    filename = suggested_name or f"upload_{uid}"
    dest_path = dest_dir / filename

    content = await file_obj.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return str(dest_path)

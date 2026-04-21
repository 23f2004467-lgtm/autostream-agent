"""Environment loading, model names, and shared file paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_PATH = DATA_DIR / "knowledge_base.md"
FAISS_CACHE_DIR = PROJECT_ROOT / ".faiss_cache"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

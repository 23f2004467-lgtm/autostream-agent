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

# The original plan targeted Gemini 1.5 Flash, but Google retired that
# model and the remaining free-tier Gemini models cap at ~20 req/day,
# which is too low to run the full test suite. We swapped to Groq's
# free tier (Llama 3.3 70B) which has identical tool-calling quality,
# a 1000 req/day budget, and 30 req/min throughput — comfortably enough
# for tests + demo + interactive use.
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

"""Pytest config: load .env + pin a slower throttle so the 14 back-to-back
test cases don't trip Groq's 30 req/min free-tier ceiling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src` importable when tests run from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set BEFORE src.llm_util is imported — that module reads the
# env var at import time into a module-level constant. Interactive use
# (Streamlit, CLI) defaults to 1.0s for snappier responses; tests use
# 3.0s so 14 cases back-to-back stay under the rate ceiling.
os.environ.setdefault("LLM_MIN_INTERVAL", "3.0")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

"""Pytest config: load .env so tests can use GOOGLE_API_KEY."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `src` importable when tests run from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

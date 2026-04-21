"""Pre-build the FAISS index for instant cold starts.

Run this once after checking out the repo, or whenever
`data/knowledge_base.md` changes. The resulting `.faiss_cache/`
directory is committed so `streamlit run app.py` never waits on
embedding work at import time.

Usage:
    python scripts/warmup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Let this script run from the repo root without an install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_community.vectorstores import FAISS  # noqa: E402

from src.config import FAISS_CACHE_DIR  # noqa: E402
from src.knowledge_base import load_chunks  # noqa: E402
from src.rag import get_embeddings  # noqa: E402


def build_index() -> None:
    chunks = load_chunks()
    print(f"[warmup] loaded {len(chunks)} KB chunks")

    embeddings = get_embeddings()
    print(f"[warmup] embedding model loaded ({type(embeddings).__name__})")

    store = FAISS.from_documents(chunks, embeddings)
    print(f"[warmup] FAISS index built with {store.index.ntotal} vectors")

    FAISS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_CACHE_DIR))
    print(f"[warmup] saved cache to {FAISS_CACHE_DIR}")
    print("[warmup] done. Commit .faiss_cache/ so cold starts stay fast.")


if __name__ == "__main__":
    build_index()

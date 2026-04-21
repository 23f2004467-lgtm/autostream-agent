"""FAISS retriever loaded from the pre-built cache.

The index is built once by `scripts/warmup.py` and committed to
`.faiss_cache/`. This module NEVER builds it on import — missing cache
is a loud error, not a silent multi-second startup.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, FAISS_CACHE_DIR


_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_VECTORSTORE: FAISS | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-init and cache the embedding model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDINGS


def load_vectorstore(cache_dir: Path | str = FAISS_CACHE_DIR) -> FAISS:
    """Load the FAISS index from the pre-built cache.

    Raises:
        FileNotFoundError: If `.faiss_cache/` is missing. Includes the
            exact command to run to rebuild.
    """
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    cache = Path(cache_dir)
    index_file = cache / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS cache not found at {cache}. "
            "Run `python scripts/warmup.py` to build it."
        )
    _VECTORSTORE = FAISS.load_local(
        str(cache),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return _VECTORSTORE


def load_retriever(k: int = 3):
    """Return a top-k cosine-similarity retriever over the KB."""
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k KB chunks and concatenate them for prompting."""
    docs = load_retriever(k=k).invoke(query)
    return "\n\n".join(d.page_content for d in docs)

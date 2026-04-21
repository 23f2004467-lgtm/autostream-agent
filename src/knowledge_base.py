"""Load and chunk the markdown knowledge base.

The chunker splits on `## ` headings so each KB section becomes one
retrieval unit. This keeps retrieval precise — a pricing question pulls
the Pro/Basic plan chunk rather than an 800-token blob.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config import KB_PATH


def load_chunks(kb_path: Path | str = KB_PATH) -> list[Document]:
    """Split the KB markdown on `## ` headings into one Document per section.

    Metadata includes the section heading so downstream code (and humans
    debugging retrieval) can see which chunk was retrieved.
    """
    text = Path(kb_path).read_text(encoding="utf-8")
    sections = text.split("\n## ")
    # First split piece contains the `# Title` header + anything before the
    # first `## ` — skip it since it's not a fact chunk.
    chunks: list[Document] = []
    for raw in sections[1:]:
        lines = raw.strip().split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        content = f"## {heading}\n{body}".strip()
        chunks.append(
            Document(page_content=content, metadata={"heading": heading})
        )
    return chunks

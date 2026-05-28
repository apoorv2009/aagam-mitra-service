from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_settings
from app.services.embedder import embed_texts
from app.services.vector_store import get_index

_PINECONE_UPSERT_BATCH = 100


def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def _chunk(text: str, *, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _stable_id(source: str, page: int, chunk_index: int) -> str:
    return hashlib.md5(f"{source}:{page}:{chunk_index}".encode()).hexdigest()


async def ingest_pdf(pdf_path: Path, filename: str) -> dict:
    settings = get_settings()
    pages = _extract_pages(pdf_path)

    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []

    for page_num, page_text in pages:
        for chunk_idx, chunk in enumerate(
            _chunk(page_text, size=settings.chunk_size_characters, overlap=settings.chunk_overlap_characters)
        ):
            texts.append(chunk)
            ids.append(_stable_id(filename, page_num, chunk_idx))
            # Store text inside metadata so Pinecone returns it at query time
            metadatas.append({"source": filename, "page": page_num, "text": chunk})

    if not texts:
        return {"file": filename, "pages": 0, "chunks": 0}

    embeddings = await embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

    index = get_index()
    vectors = [
        {"id": id_, "values": emb, "metadata": meta}
        for id_, emb, meta in zip(ids, embeddings, metadatas)
    ]
    # Upsert in batches — idempotent, re-ingesting the same PDF is safe
    for i in range(0, len(vectors), _PINECONE_UPSERT_BATCH):
        index.upsert(vectors=vectors[i : i + _PINECONE_UPSERT_BATCH])

    return {"file": filename, "pages": len(pages), "chunks": len(texts)}

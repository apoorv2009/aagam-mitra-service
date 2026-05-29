"""
Bulk ingest all PDFs from data/pdfs/ into Pinecone.

Usage:
    python ingest_all.py                  # ingest all PDFs in data/pdfs/
    python ingest_all.py path/to/file.pdf # ingest a single PDF
    python ingest_all.py path/to/folder/  # ingest all PDFs in a folder
"""

import asyncio
import sys
from pathlib import Path

from app.services.ingest import ingest_pdf


async def ingest_one(pdf_path: Path) -> None:
    print(f"  Ingesting: {pdf_path.name} ...", end=" ", flush=True)
    try:
        result = await ingest_pdf(pdf_path, pdf_path.name)
        print(f"✓  {result['pages']} pages, {result['chunks']} chunks")
    except Exception as e:
        print(f"✗  FAILED: {e}")


async def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            pdfs = sorted(target.glob("*.pdf"))
        elif target.suffix.lower() == ".pdf":
            pdfs = [target]
        else:
            print(f"Not a PDF or directory: {target}")
            return
    else:
        pdfs = sorted(Path("data/pdfs").glob("*.pdf"))

    if not pdfs:
        print("No PDFs found. Drop your PDFs in data/pdfs/ and run again.")
        return

    print(f"\nIngesting {len(pdfs)} PDF(s)...\n")
    for pdf in pdfs:
        await ingest_one(pdf)

    print("\nDone. All PDFs are now searchable via /api/v1/ask")


if __name__ == "__main__":
    asyncio.run(main())

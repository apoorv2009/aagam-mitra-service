"""
Bulk ingest all PDFs from data/jain-literature/ into Pinecone.

Usage:
    python ingest_all.py                        # ingest all PDFs in data/jain-literature/
    python ingest_all.py path/to/file.pdf       # ingest a single PDF
    python ingest_all.py path/to/folder/        # ingest all PDFs in a folder
"""

import asyncio
import sys
from pathlib import Path

# Force UTF-8 output so Hindi/Devanagari filenames print correctly on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.ingest import ingest_pdf


async def ingest_one(pdf_path: Path, retries: int = 3) -> None:
    name = pdf_path.name
    print(f"  Ingesting: {name} ...", end=" ", flush=True)
    for attempt in range(retries):
        try:
            result = await ingest_pdf(pdf_path, name)
            if result["pages"] == 0:
                print("SKIPPED  (no extractable text — may be a scanned image PDF)")
            else:
                print(f"OK  {result['pages']} pages, {result['chunks']} chunks")
            return
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"rate limited, retrying in {wait}s...", end=" ", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"FAILED: {e}")
                return


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
        pdfs = sorted(Path("data/jain-literature").glob("*.pdf"))

    if not pdfs:
        print("No PDFs found. Drop your PDFs in data/jain-literature/ and run again.")
        return

    print(f"\nIngesting {len(pdfs)} PDF(s)...\n")
    skipped = 0
    ingested = 0
    for pdf in pdfs:
        await ingest_one(pdf)

    print("\nDone.")
    print("Tip: PDFs showing 'SKIPPED' are scanned images and need OCR first.")
    print("All text-based PDFs are now searchable via /api/v1/ask")


if __name__ == "__main__":
    asyncio.run(main())

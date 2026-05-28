import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.knowledge import AskRequest, AskResponse, IngestResponse, SourceRef
from app.services.ingest import ingest_pdf
from app.services.rag import ask

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_pdf(tmp_path, file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest) -> AskResponse:
    result = await ask(req.question)
    return AskResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
    )

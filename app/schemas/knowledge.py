from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    file: str
    pages: int
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


class SourceRef(BaseModel):
    file: str | None
    page: int | None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]

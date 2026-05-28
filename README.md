# Aagam Mitra

Local AI assistant — RAG over Jain Agam texts for the temple app.

## Stack

- **Ollama** — local LLM inference (Llama 3.1 8B) and embeddings (nomic-embed-text)
- **ChromaDB** — persistent local vector store for Jain text chunks
- **FastAPI** — REST API

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/ingest` | Upload a PDF to ingest into the knowledge base |
| POST | `/api/v1/ask` | Ask a question — returns answer + source citations |

## Setup

1. Install and start [Ollama](https://ollama.com)
2. Pull required models:
   ```
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```
3. Install dependencies:
   ```
   pip install -e ".[dev]"
   ```
4. Copy `.env.example` to `.env` and adjust if needed
5. Run the service:
   ```
   uvicorn app.main:app --reload --port 8005
   ```

## Ingesting PDFs

Drop PDFs into `data/pdfs/` then POST each one:

```
curl -X POST http://localhost:8005/api/v1/ingest \
  -F "file=@data/pdfs/uttaradhyayana.pdf"
```

## Asking questions

```
curl -X POST http://localhost:8005/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Uttaradhyayana Sutra say about ahimsa?"}'
```

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openai_rag.app import ask_stream
from langchain_openai_rag.config import SUPPORTED_UPLOAD_EXTENSIONS, Settings
from langchain_openai_rag.indexing import index_uploaded_file

settings = Settings.from_env()

# src/langchain_openai_rag/api.py -> parents[2] is the project root, where the
# `static/` folder lives alongside `src/` (see Dockerfile).
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

app = FastAPI(title="LangChain RAG")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_ui() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_file), media_type="text/html")


class AskRequest(BaseModel):
    question: str


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    reset: bool = Form(True),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. Supported types: "
                + ", ".join(SUPPORTED_UPLOAD_EXTENSIONS)
            ),
        )

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        chunks_indexed = index_uploaded_file(
            tmp_path, file.filename, settings, reset=reset
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(
        content={"filename": file.filename, "chunks_indexed": chunks_indexed},
        status_code=200,
        )


@app.post("/api/ask")
async def ask_endpoint(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    async def event_stream():
        try:
            async for token in ask_stream(payload.question, settings):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:  # noqa: BLE001 - surface the error to the client
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run("langchain_openai_rag.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

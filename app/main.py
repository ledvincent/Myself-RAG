from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingest import SUPPORTED_EXTENSIONS, DocumentMetadata, ingest_file
from app.query import answer_question
from app.vectorstore import get_collection

app = FastAPI(title="RAG API", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    document_type: Literal["resume", "internship_report", "personal_bio"] = Form(...),
    language: str | None = Form(None),
    year: int | None = Form(None),
    company: str | None = Form(None),
    description: str | None = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}")
    metadata = DocumentMetadata(
        document_type=document_type,
        language=language,
        year=year,
        company=company,
        description=description,
    )
    result = ingest_file(file.filename, file.file, metadata)
    return result


@app.post("/query")
async def query(request: QueryRequest):
    collection = get_collection()
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet")
    result = answer_question(request.question, request.history)
    return result


@app.get("/sources")
async def sources():
    collection = get_collection()
    all_items = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        name = meta["source"]
        if name not in seen:
            seen[name] = {k: v for k, v in meta.items() if k != "chunk_index"}
    docs = list(seen.values())
    return {"documents": docs, "total": len(docs)}

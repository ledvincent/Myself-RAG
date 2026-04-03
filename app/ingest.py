import hashlib
import os
from typing import BinaryIO

import docx
import fitz  # pymupdf
import google.generativeai as genai

from app.vectorstore import get_collection

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"


class DocumentMetadata:
    def __init__(
        self,
        document_type: str,
        language: str | None = None,
        year: int | None = None,
        company: str | None = None,
        description: str | None = None,
    ):
        self.document_type = document_type
        self.language = language
        self.year = year
        self.company = company
        self.description = description

    def to_dict(self) -> dict:
        d = {"document_type": self.document_type}
        if self.language:
            d["language"] = self.language
        if self.year:
            d["year"] = str(self.year)
        if self.company:
            d["company"] = self.company
        if self.description:
            d["description"] = self.description
        return d


def _configure_genai() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)


def _extract_text(filename: str, file: BinaryIO) -> str:
    ext = os.path.splitext(filename)[1].lower()
    data = file.read()
    if ext == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".txt":
        return data.decode("utf-8", errors="replace")
    elif ext == ".docx":
        import io
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _embed(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
    return embeddings


def ingest_file(filename: str, file: BinaryIO, metadata: DocumentMetadata) -> dict:
    _configure_genai()
    collection = get_collection()

    text = _extract_text(filename, file)
    chunks = _chunk_text(text)

    if not chunks:
        return {"filename": filename, "chunks_ingested": 0}

    doc_id = hashlib.md5(filename.encode()).hexdigest()

    # Remove existing chunks for this document to allow re-ingestion
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    embeddings = _embed(chunks)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i, **metadata.to_dict()} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {"filename": filename, "chunks_ingested": len(chunks), "metadata": metadata.to_dict()}
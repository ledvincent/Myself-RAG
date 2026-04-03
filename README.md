# Personal RAG

A RAG system I built to answer questions about myself — my background, internships, skills — based on my own documents (resume, internship reports, personal bios in multiple languages).

The idea came from wanting a smarter way to present my profile during job applications. Instead of sending static files, anyone can just ask questions and get grounded, conversational answers.

## What it does

Upload your documents, ask questions. That's it.

Under the hood: documents are chunked, embedded with Gemini, and stored in ChromaDB. When you ask something, the most relevant chunks are retrieved and sent to Gemini as context. The model answers based only on what's in your files — no hallucination from general training data.

## Stack

- **FastAPI** — backend API
- **ChromaDB** — local vector store
- **Gemini** (`gemini-embedding-001` + `gemini-2.5-flash`) — embeddings and generation
- **PyMuPDF / python-docx** — PDF and Word parsing
- **Vanilla JS** — frontend, no framework

## Project structure

```
app/
  main.py         — endpoints (/ingest, /query, /sources)
  ingest.py       — parsing, chunking, embedding
  query.py        — retrieval + generation
  vectorstore.py  — ChromaDB client
  static/
    index.html    — web UI
start.py          — starts server and opens browser
```

## Getting started

You'll need Python 3.12 and a [Gemini API key](https://aistudio.google.com/app/apikey).

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

conda create -n rag python=3.12
conda activate rag
pip install -r requirements.txt

cp .env.example .env
# add your GEMINI_API_KEY to .env

python start.py
```

Run the *start.bat* file to host a local website, upload your documents, start asking.

## Docker

```bash
docker build -t personal-rag .
docker run -p 8000:8000 --env-file .env -v ./chroma_data:/app/chroma_data personal-rag
```

## A few things to know

- `chroma_data/` and `.env` are both gitignored — your documents and API key stay local
- The knowledge base is file-based and local by design; swapping to a cloud vector DB (Pinecone, Qdrant) would make it shareable without changing anything else

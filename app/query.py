import os

import google.generativeai as genai

from app.vectorstore import get_collection

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 5

SYSTEM_INSTRUCTION = """\
You are a knowledgeable assistant answering questions about a specific person based on their personal documents.
Write in a natural, fluent tone as if you simply know this information.
Do NOT cite filenames, document names, or say where each fact comes from.
Answer directly and coherently. If the context does not contain enough information, say so clearly."""

CONTEXT_TEMPLATE = """\
Here is relevant context retrieved from the person's documents to help you answer:

{context}

Question: {question}"""

MAX_HISTORY_TURNS = 10


def _configure_genai() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)


def _embed_query(question: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=question,
        task_type="retrieval_query",
    )
    return result["embedding"]


def answer_question(question: str, history: list = []) -> dict:
    _configure_genai()
    collection = get_collection()

    query_embedding = _embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_blocks = []
    for chunk, meta in zip(chunks, metadatas):
        meta_parts = [f"type={meta.get('document_type', 'unknown')}"]
        if meta.get("language"):
            meta_parts.append(f"language={meta['language']}")
        if meta.get("year"):
            meta_parts.append(f"year={meta['year']}")
        if meta.get("company"):
            meta_parts.append(f"company={meta['company']}")
        if meta.get("description"):
            meta_parts.append(f"description={meta['description']}")
        header = f"[Source: {meta.get('source', 'unknown')} | {', '.join(meta_parts)}]"
        context_blocks.append(f"{header}\n{chunk}")

    context = "\n\n---\n\n".join(context_blocks)
    user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

    # Cap history to last N turns to control token usage
    recent_history = history[-MAX_HISTORY_TURNS * 2:]
    chat_history = [
        {"role": "user" if msg.role == "user" else "model", "parts": [msg.content]}
        for msg in recent_history
    ]

    model = genai.GenerativeModel(
        GENERATION_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(user_message)

    sources = [
        {
            "source": m["source"],
            "chunk_index": m["chunk_index"],
            "similarity": round(1 - d, 4),
        }
        for m, d in zip(metadatas, distances)
    ]

    return {
        "question": question,
        "answer": response.text,
        "sources": sources,
    }

"""
rag_core.py
Core RAG pipeline logic — shared by the Streamlit app.
Separation of concerns: loader, chunker, embedder, vector_store, retriever, generator.
"""

import os
import re
import time
import uuid
import logging
from typing import List, Dict, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="rag_system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("RAG_CORE")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 20
INDEX_NAME = "intermediate-rag-index"
NAMESPACE = "rag-session"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MESSAGE = "The answer is not available in the provided document."

STRICT_SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "You must answer the user's question using ONLY the CONTEXT provided below. "
    "Do not use any outside knowledge. Do not guess or infer beyond what is written. "
    f'If the context does not contain enough information to answer, respond exactly with: "{FALLBACK_MESSAGE}" '
    "Keep answers concise and cite which page(s) you used inline, e.g. (Page 3)."
)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_embedder = None
_pc_client = None
_pinecone_index = None
_groq_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_embed_dim():
    return get_embedder().get_sentence_embedding_dimension()


def init_pinecone(api_key: str):
    global _pc_client, _pinecone_index
    _pc_client = Pinecone(api_key=api_key)
    existing = [idx["name"] for idx in _pc_client.list_indexes()]
    if INDEX_NAME not in existing:
        _pc_client.create_index(
            name=INDEX_NAME,
            dimension=get_embed_dim(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not _pc_client.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    _pinecone_index = _pc_client.Index(INDEX_NAME)
    return _pinecone_index


def get_index():
    if _pinecone_index is None:
        raise ConnectionError("Pinecone index not initialized. Call init_pinecone() first.")
    return _pinecone_index


def init_groq(api_key: str):
    global _groq_client
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def get_groq():
    if _groq_client is None:
        raise ConnectionError("Groq client not initialized. Call init_groq() first.")
    return _groq_client


def clear_namespace():
    """Deletes all vectors in the current namespace — useful before re-ingesting to avoid duplicates."""
    get_index().delete(delete_all=True, namespace=NAMESPACE)
    logger.info(f"Namespace '{NAMESPACE}' cleared.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_pdf(filename: str, size_bytes: int):
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"Invalid file type: '{filename}'. Only PDF files are accepted.")
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"'{filename}' is {size_mb:.2f} MB — exceeds the {MAX_FILE_SIZE_MB} MB limit.")
    return True


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n(?=\w)", "", text)
    return text.strip()


def extract_text_from_pdf(doc_name: str, pdf_bytes: bytes) -> List[Dict]:
    pages_data = []
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open '{doc_name}' — invalid or corrupted PDF. ({e})")

    if pdf.page_count == 0:
        raise ValueError(f"'{doc_name}' has no pages.")

    for page_num in range(pdf.page_count):
        raw_text = pdf[page_num].get_text("text")
        cleaned = clean_text(raw_text)
        if cleaned:
            pages_data.append({"doc_name": doc_name, "page": page_num + 1, "text": cleaned})

    pdf.close()
    if not pages_data:
        raise ValueError(f"No extractable text found in '{doc_name}' (likely scanned/image-only).")

    return pages_data


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_pages(pages_data: List[Dict], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(page_chunks):
            chunk_id = f"{page['doc_name']}_p{page['page']}_c{i}_{uuid.uuid4().hex[:6]}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": page["doc_name"],
                "page": page["page"],
                "text": chunk_text,
            })
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + upsert
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def upsert_chunks(chunks: List[Dict], batch_size: int = 64):
    if not chunks:
        raise ValueError("No chunks to upsert.")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    to_upsert = [{
        "id": c["chunk_id"],
        "values": v,
        "metadata": {"doc_name": c["doc_name"], "page": c["page"], "chunk_id": c["chunk_id"], "text": c["text"]},
    } for c, v in zip(chunks, vectors)]

    index = get_index()
    for i in range(0, len(to_upsert), batch_size):
        index.upsert(vectors=to_upsert[i:i + batch_size], namespace=NAMESPACE)

    return len(to_upsert)


def ingest_pdf(doc_name: str, pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> Dict:
    validate_pdf(doc_name, len(pdf_bytes))
    pages_data = extract_text_from_pdf(doc_name, pdf_bytes)
    chunks = chunk_pages(pages_data, chunk_size, chunk_overlap)
    upsert_chunks(chunks)
    logger.info(f"Ingested {doc_name}: {len(pages_data)} pages, {len(chunks)} chunks.")
    return {"doc_name": doc_name, "pages": len(pages_data), "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve_context(query: str, top_k: int = 5, similarity_threshold: float = 0.3,
                      filter_doc: Optional[str] = None, filter_page: Optional[int] = None) -> List[Dict]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    query_vector = embed_texts([query])[0]
    pinecone_filter = {}
    if filter_doc:
        pinecone_filter["doc_name"] = {"$eq": filter_doc}
    if filter_page:
        pinecone_filter["page"] = {"$eq": filter_page}

    try:
        results = get_index().query(
            vector=query_vector,
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None,
        )
    except Exception as e:
        raise ConnectionError(f"Pinecone connection/query failed: {e}")

    matches = []
    for m in results.get("matches", []):
        if m["score"] >= similarity_threshold:
            matches.append({
                "score": round(m["score"], 4),
                "doc_name": m["metadata"]["doc_name"],
                "page": m["metadata"]["page"],
                "text": m["metadata"]["text"],
            })
    return matches


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def build_prompt(query: str, matches: List[Dict]) -> str:
    blocks = [f"[{m['doc_name']} - Page {m['page']}]\n{m['text']}" for m in matches]
    return f"CONTEXT:\n{'---'.join(blocks)}\n\nQUESTION: {query}\n\nANSWER:"


def generate_answer(query: str, matches: List[Dict]) -> str:
    if not matches:
        return FALLBACK_MESSAGE
    prompt = build_prompt(query, matches)
    try:
        response = get_groq().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        return f"LLM generation error: {e}"


def confidence_from_matches(matches: List[Dict]) -> Dict:
    if not matches:
        return {"pct": 0, "label": "None"}
    avg = sum(m["score"] for m in matches) / len(matches)
    pct = round(avg * 100, 1)
    label = "High" if pct >= 75 else "Medium" if pct >= 50 else "Low"
    return {"pct": pct, "label": label}


def ask_question(query: str, top_k=5, similarity_threshold=0.3, filter_doc=None, filter_page=None) -> Dict:
    matches = retrieve_context(query, top_k, similarity_threshold, filter_doc, filter_page)
    answer = generate_answer(query, matches)
    confidence = confidence_from_matches(matches)
    logger.info(f"Q: {query} | Confidence: {confidence['pct']}% | Sources: {len(matches)}")
    return {"query": query, "answer": answer, "confidence": confidence, "sources": matches}

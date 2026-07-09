# Intermediate RAG System Using Pinecone Vector Database

> **Course:** Artificial Intelligence / NLP / Applied LLM Systems
> **Level:** Intermediate
> **Type:** Internship Assignment — Task 2

A Retrieval-Augmented Generation (RAG) system that answers questions **strictly from an uploaded PDF document**. It extracts and chunks document text, generates vector embeddings, stores them in **Pinecone**, retrieves the most relevant chunks for a query, and generates a grounded, traceable answer using **Groq (Llama 3.3 70B)** — with hallucination prevention built into every layer.

---

## 🎯 Objective

Design and implement a RAG system that:
- Accepts a PDF document as input
- Extracts and processes text from the document
- Generates embeddings
- Stores embeddings in a Pinecone vector database
- Retrieves relevant context based on a user query
- Generates accurate answers **strictly** from the PDF content
- Prevents hallucination and provides traceable, source-attributed answers

---

## 🏗️ System Architecture
PDF Upload
↓
Text Extraction        (PyMuPDF)
↓
Text Chunking           (LangChain RecursiveCharacterTextSplitter)
↓
Embedding Generation     (Sentence-Transformers: all-MiniLM-L6-v2)
↓
Pinecone Vector Indexing  (Serverless Index + Namespace + Metadata)
↓
Semantic Retrieval        (Cosine similarity, top-k, threshold, metadata filter)
↓
LLM Response Generation    (Groq — Llama 3.3 70B, strict context-only prompt)
↓
Answer + Source Reference   (Page, excerpt, similarity score, confidence)
📎 See [`docs/architecture_diagram.png`](docs/architecture_diagram.png) for the full visual diagram.

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Interface | Streamlit (custom-styled) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector Database | Pinecone (Serverless, AWS `us-east-1`, cosine metric) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Config | `.env` file via `python-dotenv` / `getpass` |

---

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/intermediate-rag-pinecone-groq.git
cd intermediate-rag-pinecone-groq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
Copy the example env file and fill in your real keys:
```bash
cp .env.example .env
```
Then edit `.env`:
- Get a Pinecone key: https://app.pinecone.io → API Keys
- Get a Groq key: https://console.groq.com/keys

### 4. Run the app

**Option A — Streamlit app (recommended)**
```bash
streamlit run app/app.py
```

**Option B — Jupyter/Colab notebook**
Open `notebook/Intermediate_RAG_Pinecone_Groq.ipynb` and run cells top to bottom.

---

## 🔹 Pinecone Configuration

| Setting | Value |
|---|---|
| Index name | `intermediate-rag-index` |
| Metric | `cosine` |
| Dimension | `384` |
| Spec | Serverless — `aws`, `us-east-1` |
| Namespace | `rag-session` |
| Metadata fields | `doc_name`, `page`, `chunk_id`, `text` |

---

## 📈 Enhancements Implemented (≥3 required)

- ✅ **Multi-document support** — upload and query across several PDFs
- ✅ **Query history (session memory)** — every Q&A logged in-session
- ✅ **Adjustable chunk size** — configurable from the sidebar UI
- ✅ **Adjustable top-k retrieval** — configurable from the sidebar UI
- ✅ **Metadata filtering** — filter by document name and/or page number
- ✅ **Confidence scoring display** — derived from average similarity of top matches
- ✅ **Logging user queries** — written to `rag_system.log`

---

## 🛡️ Hallucination Prevention

- A **similarity threshold** filters out weakly related chunks before they reach the LLM.
- If **no chunk** clears the threshold, the LLM is never called — the system returns the fixed fallback message directly:
  > "The answer is not available in the provided document."
- The LLM system prompt strictly instructs the model to answer **only** from the provided context.
- Every answer includes **verifiable sources** (document, page, excerpt, similarity score) for human auditing.

---

## ⚠️ Known Limitations

- Abstract/meta-level questions (e.g. "what is the main topic of this document?") tend to score lower on cosine similarity than specific factual questions, since retrieval matches content similarity rather than document-level intent. A dedicated "document overview" chunk could improve this in future iterations.
- Scanned/image-only PDFs are not supported since no text layer exists to extract (OCR is a possible future enhancement).
- The current setup uses a single shared namespace per session; a production version would isolate namespaces per user/session for true multi-tenancy.

---

## 👤 Author
Saira Ejaz
Submitted as part of an AI/NLP internship task — Intermediate RAG System Using Pinecone Vector Database.

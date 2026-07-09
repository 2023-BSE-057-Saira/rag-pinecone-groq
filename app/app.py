"""
app.py
Streamlit frontend for the Intermediate RAG System (Pinecone + Groq).
Run with: streamlit run app.py
"""

import streamlit as st
from datetime import datetime
import rag_core as core

st.set_page_config(page_title="DocuMind", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace; }

    :root {
        --bg: #0a0a0d;
        --surface: #131317;
        --surface-2: #1a1a20;
        --border: rgba(255,255,255,0.08);
        --border-soft: rgba(255,255,255,0.05);
        --text: #f4f4f5;
        --text-muted: #9a9aa5;
        --text-faint: #5f5f6b;
        --accent: #5b6cff;
        --accent-soft: rgba(91,108,255,0.12);
        --good: #34d399;
        --warn: #fbbf24;
        --bad: #f87171;
    }

    .stApp { background: var(--bg); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }

    .app-header {
        display: flex; align-items: center; gap: 14px;
        padding-bottom: 20px; margin-bottom: 28px;
        border-bottom: 1px solid var(--border-soft);
    }
    .app-logo {
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(135deg, #5b6cff, #8b7bff);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; font-weight: 800; color: white;
        box-shadow: 0 4px 20px rgba(91,108,255,0.35);
        flex-shrink: 0;
    }
    .app-title { font-size: 20px; font-weight: 700; color: var(--text); line-height: 1.2; }
    .app-subtitle { font-size: 13px; color: var(--text-muted); margin-top: 2px; }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .panel-title {
        font-size: 15px; font-weight: 700; color: var(--text);
        margin-bottom: 4px; display: flex; align-items: center; gap: 8px;
    }
    .panel-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 18px; }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    .sb-section-title {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--text-faint);
        margin: 20px 0 10px 0;
    }
    .sb-section-title:first-child { margin-top: 0; }

    .status-row { display: flex; gap: 8px; margin-bottom: 4px; }
    .status-pill {
        flex: 1; padding: 8px 10px; border-radius: 8px; font-size: 12px;
        font-weight: 600; text-align: center; border: 1px solid var(--border);
    }
    .status-on { background: rgba(52,211,153,0.1); color: var(--good); border-color: rgba(52,211,153,0.25); }
    .status-off { background: rgba(255,255,255,0.03); color: var(--text-faint); }

    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 9px 18px !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        box-shadow: none !important;
        transition: filter 0.15s ease, transform 0.15s ease !important;
        width: 100%;
    }
    .stButton > button:hover { filter: brightness(1.12); transform: translateY(-1px); }
    .stButton > button:active { transform: translateY(0); }

    .stTextInput input, .stNumberInput input {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
        color: var(--text) !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--surface); padding: 5px;
        border-radius: 11px; border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 18px; font-weight: 600; font-size: 13.5px;
        color: var(--text-muted); background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface-2) !important; color: var(--text) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: 12px !important;
    }

    .doc-row {
        display: flex; align-items: center; justify-content: space-between;
        background: var(--surface-2); border: 1px solid var(--border-soft);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .doc-name { font-weight: 600; font-size: 13.5px; color: var(--text); }
    .doc-meta { font-size: 12px; color: var(--text-muted); }

    .chat-question {
        background: var(--accent-soft);
        border: 1px solid rgba(91,108,255,0.25);
        border-radius: 12px 12px 4px 12px;
        padding: 13px 16px; margin-bottom: 10px;
        color: var(--text); font-size: 14.5px; font-weight: 500;
        max-width: 85%; margin-left: auto;
    }
    .chat-answer {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px 12px 12px 4px;
        padding: 16px 18px; margin-bottom: 6px;
        color: var(--text); font-size: 14.5px; line-height: 1.65;
        max-width: 85%;
    }

    .conf-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 7px; font-size: 12px; font-weight: 700;
        margin-bottom: 16px;
    }
    .conf-dot { width: 7px; height: 7px; border-radius: 50%; }
    .conf-high { background: rgba(52,211,153,0.1); color: var(--good); }
    .conf-high .conf-dot { background: var(--good); }
    .conf-medium { background: rgba(251,191,36,0.1); color: var(--warn); }
    .conf-medium .conf-dot { background: var(--warn); }
    .conf-low, .conf-none { background: rgba(248,113,113,0.1); color: var(--bad); }
    .conf-low .conf-dot, .conf-none .conf-dot { background: var(--bad); }

    .source-label {
        font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--text-faint); margin: 18px 0 10px 0;
    }
    .source-card {
        background: var(--surface-2); border: 1px solid var(--border-soft);
        border-left: 3px solid var(--accent);
        border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 8px;
    }
    .source-meta {
        display: flex; align-items: center; gap: 10px;
        font-size: 11.5px; color: var(--accent); font-weight: 700; margin-bottom: 6px;
    }
    .source-score {
        background: rgba(91,108,255,0.12); padding: 2px 8px; border-radius: 5px;
        font-size: 11px; color: var(--accent);
    }
    .source-excerpt { font-size: 13px; color: var(--text-muted); line-height: 1.5; font-style: italic; }

    .metric-box {
        background: var(--surface-2); border: 1px solid var(--border-soft);
        border-radius: 10px; padding: 14px; text-align: center;
    }
    .metric-num { font-size: 24px; font-weight: 800; color: var(--text); line-height: 1; }
    .metric-lbl { font-size: 11px; color: var(--text-muted); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.04em; }

    .empty-state {
        text-align: center; padding: 50px 20px; color: var(--text-faint);
    }
    .empty-state-icon { font-size: 32px; margin-bottom: 10px; opacity: 0.5; }

    hr { border-color: var(--border-soft); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {"documents": [], "query_history": [], "pinecone_ready": False, "groq_ready": False}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="app-header">
    <div class="app-logo">◆</div>
    <div>
        <div class="app-title">DocuMind</div>
        <div class="app-subtitle">Grounded document Q&A · Pinecone + Groq</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sb-section-title">Connection</div>', unsafe_allow_html=True)

    p_status = "status-on" if st.session_state.pinecone_ready else "status-off"
    g_status = "status-on" if st.session_state.groq_ready else "status-off"
    st.markdown(f"""
    <div class="status-row">
        <div class="status-pill {p_status}">Pinecone {'●' if st.session_state.pinecone_ready else '○'}</div>
        <div class="status-pill {g_status}">Groq {'●' if st.session_state.groq_ready else '○'}</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    pinecone_key = st.text_input("Pinecone API Key", type="password", label_visibility="collapsed", placeholder="Pinecone API Key")
    groq_key = st.text_input("Groq API Key", type="password", label_visibility="collapsed", placeholder="Groq API Key")

    if st.button("Connect Services"):
        try:
            with st.spinner("Connecting..."):
                core.init_pinecone(pinecone_key)
                st.session_state.pinecone_ready = True
                core.init_groq(groq_key)
                st.session_state.groq_ready = True
            st.rerun()
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.markdown('<div class="sb-section-title">Ingestion</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size", 300, 2000, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 120, step=20)

    st.markdown('<div class="sb-section-title">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K results", 1, 15, 5)
    similarity_threshold = st.slider("Min similarity", 0.0, 0.9, 0.3, step=0.05)

    st.markdown('<div class="sb-section-title">Filters</div>', unsafe_allow_html=True)
    filter_doc = st.selectbox("Document", options=["All documents"] + [d["doc_name"] for d in st.session_state.documents], label_visibility="collapsed")
    filter_page = st.number_input("Page (0 = any)", min_value=0, value=0, step=1, label_visibility="collapsed")

    st.markdown('<div class="sb-section-title">Session</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-num">{len(st.session_state.documents)}</div><div class="metric-lbl">Docs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-num">{len(st.session_state.query_history)}</div><div class="metric-lbl">Queries</div></div>', unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab_upload, tab_ask, tab_history = st.tabs(["Upload", "Ask", "History"])

# ---------------------------------------------------------------------------
with tab_upload:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📄 Upload documents</div>
        <div class="panel-desc">PDF only, up to 20 MB each. Multiple files supported.</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    process_clicked = st.button("Process & Index")

    if process_clicked:
        if not st.session_state.pinecone_ready:
            st.error("Connect to Pinecone first, in the sidebar.")
        elif not uploaded_files:
            st.warning("Upload at least one PDF.")
        else:
            progress = st.progress(0, text="Starting...")
            for i, f in enumerate(uploaded_files):
                try:
                    result = core.ingest_pdf(f.name, f.read(), chunk_size, chunk_overlap)
                    st.session_state.documents.append(result)
                except ValueError as e:
                    st.error(str(e))
                progress.progress((i + 1) / len(uploaded_files), text=f"Processed {f.name}")
            progress.empty()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.documents:
        st.markdown('<div class="panel"><div class="panel-title">Indexed documents</div>', unsafe_allow_html=True)
        for d in st.session_state.documents:
            st.markdown(f"""
            <div class="doc-row">
                <div class="doc-name">📎 {d['doc_name']}</div>
                <div class="doc-meta">{d['pages']} pages · {d['chunks']} chunks</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
with tab_ask:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Ask", placeholder="Ask something about your document...", label_visibility="collapsed")
    with col2:
        ask_clicked = st.button("Ask")
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked:
        if not st.session_state.groq_ready or not st.session_state.pinecone_ready:
            st.error("Connect to Pinecone and Groq first, in the sidebar.")
        elif not query.strip():
            st.warning("Type a question first.")
        else:
            fdoc = None if filter_doc == "All documents" else filter_doc
            fpage = filter_page if filter_page > 0 else None
            try:
                with st.spinner("Thinking..."):
                    result = core.ask_question(query, top_k=top_k, similarity_threshold=similarity_threshold,
                                                filter_doc=fdoc, filter_page=fpage)
                result["timestamp"] = str(datetime.now())
                st.session_state.query_history.append(result)
            except (ValueError, ConnectionError) as e:
                st.error(str(e))

    if st.session_state.query_history:
        latest = st.session_state.query_history[-1]
        st.markdown(f'<div class="chat-question">{latest["query"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-answer">{latest["answer"]}</div>', unsafe_allow_html=True)

        conf = latest["confidence"]
        conf_class = f"conf-{conf['label'].lower()}"
        st.markdown(f"""
        <div class="conf-badge {conf_class}">
            <span class="conf-dot"></span> {conf['pct']}% confidence · {conf['label']}
        </div>
        """, unsafe_allow_html=True)

        if latest["sources"]:
            st.markdown('<div class="source-label">Sources</div>', unsafe_allow_html=True)
            for s in latest["sources"]:
                excerpt = (s["text"][:200] + "…") if len(s["text"]) > 200 else s["text"]
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-meta">
                        📄 {s['doc_name']} · Page {s['page']}
                        <span class="source-score">{s['score']}</span>
                    </div>
                    <div class="source-excerpt">"{excerpt}"</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            Ask your first question to get started.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
with tab_history:
    st.markdown('<div class="panel"><div class="panel-title">Query history</div>', unsafe_allow_html=True)

    if not st.session_state.query_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🗂️</div>
            No queries yet this session.
        </div>
        """, unsafe_allow_html=True)
    else:
        for rec in reversed(st.session_state.query_history):
            conf = rec["confidence"]
            with st.expander(f"{rec['query']}  ·  {conf['pct']}% confidence"):
                st.markdown(f'<div class="chat-answer" style="max-width:100%;">{rec["answer"]}</div>', unsafe_allow_html=True)
                st.caption(f"{rec['timestamp'][:19]} · {len(rec['sources'])} sources")

    st.markdown("</div>", unsafe_allow_html=True)

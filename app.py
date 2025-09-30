import streamlit as st
import os
import zipfile
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

# Try FAISS, show a helpful message if not installed
try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    faiss = None

import requests

# ------------- App Config and Styling -------------
st.set_page_config(page_title="VectorInsight", page_icon="📁", layout="wide")

# Subtle CSS to polish visuals
st.markdown("""
<style>
    .app-header { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem; }
    .muted { color: #6b7280; }
    .card { padding: 0.85rem 1rem; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }
    .chunk { background: #f9fafb; border: 1px solid #e5e7eb; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; white-space: pre-wrap; }
    .metric-box { text-align: center; padding: 0.5rem 0; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }
    .footer { color: #6b7280; font-size: 0.85rem; margin-top: 1.25rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header">📁 VectorInsight</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Multi-file PDF/TXT RAG with LLaMA 3</div>', unsafe_allow_html=True)
st.divider()

# ------------- Sidebar Controls -------------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3, help="How many chunks to retrieve for context")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1200, step=50, value=500)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, step=20, value=80)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    st.caption("Secret")
    if "GROQ_API_KEY" in st.secrets:
        st.success("GROQ_API_KEY found")
    else:
        st.warning("GROQ_API_KEY missing in st.secrets")

    st.button("Clear Session", on_click=lambda: st.session_state.clear())

# ------------- Utility: Text Splitter -------------
def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "!", "?", ";", ","]
    )
    return splitter.split_text(text)

# ------------- File Parsing -------------
def extract_text_from_file(file, filetype: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (text, metadata)
    metadata keys: pages (for PDF), chars, name (if available)
    """
    name = getattr(file, "name", None)
    if filetype.lower() == ".pdf":
        # For UploadedFile, reading consumes stream; use bytes to open safely
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        pages = doc.page_count
        text = "\n".join(page.get_text() for page in doc)
        return text, {"pages": pages, "chars": len(text), "name": name}
    elif filetype.lower() == ".txt":
        data = file.read()
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = str(data)
        return text, {"pages": None, "chars": len(text), "name": name}
    return "", {"pages": None, "chars": 0, "name": name}

def extract_text_from_zip(uploaded_zip) -> Tuple[str, List[Dict[str, Any]]]:
    """Returns full_text, list of per-file metadata dicts."""
    extracted_texts = []
    metas: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmpdir)
        for file in Path(tmpdir).rglob("*"):
            if file.is_file() and file.suffix.lower() in [".pdf", ".txt"]:
                with open(file, "rb") as f:
                    text, meta = extract_text_from_file(f, file.suffix)
                    meta["name"] = str(file.name)
                    extracted_texts.append(text)
                    metas.append(meta)
    return "\n".join(extracted_texts), metas

# ------------- Embeddings / Index -------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(chunks: List[str], model) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine with normalized vectors ~ inner product
    index.add(embeddings)
    return index

def search_index(query: str, model, index, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, k)
    return distances[0], indices[0]

# ------------- LLM -------------
def generate_llama3_answer(query: str, context: str, temperature: float = 0.3) -> str:
    if "GROQ_API_KEY" not in st.secrets:
        raise RuntimeError("GROQ_API_KEY is missing in st.secrets")

    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer clearly. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
    }

    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError(f"LLaMA 3 API Error: {resp.status_code} {resp.text}")

# ------------- Session State -------------
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []             # List[str]
if "chunk_meta" not in st.session_state:
    st.session_state["chunk_meta"] = []         # parallel list with source/name, offsets
if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None
if "index" not in st.session_state:
    st.session_state["index"] = None
if "model" not in st.session_state:
    st.session_state["model"] = load_model()

# ------------- UI: Uploader and Processing -------------
uploaded_files = st.file_uploader(
    "Upload PDF/TXT/ZIP files",
    type=["pdf", "txt", "zip"],
    accept_multiple_files=True
)

query = st.text_input("Ask a question about the uploaded documents")

# Show file summary
if uploaded_files:
    total_files = len(uploaded_files)
    total_pages = 0
    total_chars = 0

    with st.spinner("Reading files..."):
        full_texts: List[str] = []
        meta_list: List[Dict[str, Any]] = []

        for file in uploaded_files:
            if file.name.lower().endswith(".zip"):
                text, metas = extract_text_from_zip(file)
                full_texts.append(text)
                meta_list.extend(metas)
            elif file.name.lower().endswith((".pdf", ".txt")):
                # We need to call read() once; Streamlit UploadedFile is a buffer. Reset cursor after read.
                text, meta = extract_text_from_file(file, Path(file.name).suffix)
                full_texts.append(text)
                meta["name"] = file.name
                meta_list.append(meta)
            else:
                st.warning(f"Unsupported file type: {file.name}")

        total_pages = sum((m.get("pages") or 0) for m in meta_list)
        total_chars = sum((m.get("chars") or 0) for m in meta_list)

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-box"><div class="muted">Files</div><h3>{total_files}</h3></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="muted">PDF Pages</div><h3>{total_pages}</h3></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="muted">Characters</div><h3>{total_chars:,}</h3></div>', unsafe_allow_html=True)

    with st.expander("View file details"):
        for m in meta_list:
            st.markdown(f"- {m.get('name','(unknown)')} • pages: {m.get('pages') or 0} • chars: {m.get('chars') or 0}")

    # Process button to chunk + index
    if st.button("🔍 Process Documents", type="primary"):
        all_text = "\n".join(full_texts).strip()
        if not all_text:
            st.error("No valid text extracted from the uploaded files.")
        else:
            with st.spinner("Chunking and embedding..."):
                chunks = chunk_text(all_text, size=chunk_size, overlap=chunk_overlap)
                # Build parallel metadata that at least shows source as 'combined'
                chunk_meta = [{"source": "uploaded", "id": i} for i in range(len(chunks))]
                model = st.session_state["model"]
                embeddings = embed_text(chunks, model)
                try:
                    index = build_faiss_index(embeddings)
                except Exception as e:
                    st.error(str(e))
                    index = None

                st.session_state["chunks"] = chunks
                st.session_state["chunk_meta"] = chunk_meta
                st.session_state["embeddings"] = embeddings
                st.session_state["index"] = index

            if st.session_state["index"] is not None:
                st.success(f"Processed {len(chunks)} chunks. Ready to query!")
            else:
                st.error("Index not built. Install FAISS (faiss-cpu) and retry.")

else:
    st.info("Upload at least one .pdf, .txt, or .zip file to get started.")

# ------------- Query + Results -------------
if query and st.session_state.get("index") is not None:
    with st.spinner("Searching and querying LLaMA 3..."):
        try:
            distances, idxs = search_index(query, st.session_state["model"], st.session_state["index"], k)
            chunks = st.session_state["chunks"]
            selected = [(i, chunks[i], float(distances[pos])) for pos, i in enumerate(idxs) if i < len(chunks)]
            context = "\n\n".join([c for _, c, _ in selected])
            answer = generate_llama3_answer(query, context, temperature=temperature)
        except Exception as e:
            answer = None
            st.error(str(e))

    if answer is not None:
        tabs = st.tabs(["🤖 Answer", "📚 Sources", "🧩 Chunks"])
        with tabs[0]:
            st.subheader("Answer")
            st.write(f"Q: {query}")
            st.write(answer)
        with tabs[1]:
            st.subheader("Top Matching Chunks (as context)")
            for rank, (i, chunk, score) in enumerate(selected, start=1):
                with st.expander(f"Chunk #{i} • Score: {score:.3f}", expanded=(rank == 1)):
                    st.markdown(f"<div class='chunk'>{chunk}</div>", unsafe_allow_html=True)
        with tabs[2]:
            st.subheader("All Retrieved Chunks")
            for rank, (i, chunk, score) in enumerate(selected, start=1):
                st.markdown(f"**{rank}. Index {i}** — score {score:.3f}")
                st.code(chunk)

elif query:
    st.info("Please process documents first, then ask your question.")

st.markdown('<div class="footer">Tip: If you see FAISS errors on Windows, install CPU-only with <code>pip install faiss-cpu</code>.</div>', unsafe_allow_html=True)

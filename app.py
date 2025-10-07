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
import requests

try:
    import faiss
except Exception:
    faiss = None

# ------------------ App Config ------------------
st.set_page_config(page_title="VectorInsight Chat", page_icon="💬", layout="wide")

st.markdown("""
<style>
.chat-container {
    background: #fafafa;
    border-radius: 12px;
    padding: 1rem;
    max-height: 550px;
    overflow-y: auto;
}
.user-msg {
    background: #DCF8C6;
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    max-width: 80%;
    align-self: flex-end;
}
.bot-msg {
    background: #FFF;
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    border: 1px solid #e5e7eb;
    max-width: 80%;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 VectorInsight — Chat • Doc-QA • Summariser")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Select Mode", ["Chat", "Doc-QA", "Summariser"], horizontal=True)
    k = st.slider("Top-K Chunks", 1, 10, 3)
    chunk_size = st.slider("Chunk Size", 200, 1200, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 80, 20)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.button("🧹 Clear Session", on_click=lambda: st.session_state.clear())

# ------------------ Helper Functions ------------------
def chunk_text(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "!", "?", ";", ","]
    )
    return splitter.split_text(text)

def extract_text_from_file(file, ext):
    name = getattr(file, "name", None)
    if ext.lower() == ".pdf":
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text, {"pages": doc.page_count, "chars": len(text), "name": name}
    elif ext.lower() == ".txt":
        data = file.read()
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = str(data)
        return text, {"pages": None, "chars": len(text), "name": name}
    return "", {"pages": None, "chars": 0, "name": name}

def extract_text_from_zip(uploaded_zip):
    texts, metas = [], []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmpdir)
        for file in Path(tmpdir).rglob("*"):
            if file.suffix.lower() in [".pdf", ".txt"]:
                with open(file, "rb") as f:
                    text, meta = extract_text_from_file(f, file.suffix)
                    meta["name"] = file.name
                    texts.append(text)
                    metas.append(meta)
    return "\n".join(texts), metas

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(chunks, model):
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(query, model, index, k):
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(q_vec, k)
    return distances[0], indices[0]

def llama3_generate(prompt, temperature=0.3):
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Missing GROQ_API_KEY in secrets")
        return ""
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"API Error: {r.text}")
        return ""

# ------------------ Session Init ------------------
if "model" not in st.session_state:
    st.session_state["model"] = load_model()
if "history" not in st.session_state:
    st.session_state["history"] = []
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "index" not in st.session_state:
    st.session_state["index"] = None
if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""

# ------------------ Upload ------------------
uploaded_files = st.file_uploader("📂 Upload PDF/TXT/ZIP files", type=["pdf", "txt", "zip"], accept_multiple_files=True)
if uploaded_files:
    full_texts, metas = [], []
    for f in uploaded_files:
        if f.name.endswith(".zip"):
            text, m = extract_text_from_zip(f)
            full_texts.append(text)
            metas += m
        else:
            text, meta = extract_text_from_file(f, Path(f.name).suffix)
            full_texts.append(text)
            metas.append(meta)
    combined_text = "\n".join(full_texts)
    st.session_state["uploaded_text"] = combined_text

    if mode in ["Doc-QA", "Summariser"] and st.button("⚙️ Process Documents"):
        with st.spinner("Chunking and embedding..."):
            chunks = chunk_text(combined_text, chunk_size, chunk_overlap)
            embeddings = embed_text(chunks, st.session_state["model"])
            index = build_faiss_index(embeddings)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
        st.success(f"Indexed {len(chunks)} chunks ✅")

# ------------------ Chat Interface ------------------
st.markdown("### 💭 Chat Window")
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["history"]:
        role, text = msg
        css = "user-msg" if role == "user" else "bot-msg"
        st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["history"].append(("user", user_input))
    with st.spinner("Thinking..."):
        if mode == "Chat":
            prompt = user_input
        elif mode == "Doc-QA":
            if st.session_state["index"] is None:
                st.warning("Please process documents first.")
                prompt = "Please process documents first."
            else:
                dists, idxs = search_index(user_input, st.session_state["model"], st.session_state["index"], k)
                context = "\n\n".join([st.session_state["chunks"][i] for i in idxs])
                prompt = f"""You are a helpful assistant. 
Use ONLY the context below to answer clearly. 
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {user_input}
Answer:"""
        elif mode == "Summariser":
            if not st.session_state["uploaded_text"]:
                st.warning("Upload documents first.")
                prompt = "Please upload documents first."
            else:
                prompt = f"Summarise the following document clearly and concisely:\n\n{st.session_state['uploaded_text']}"
        answer = llama3_generate(prompt, temperature)
        st.session_state["history"].append(("bot", answer))

    st.rerun()

st.markdown("<br><sub>🧠 VectorInsight Multi-Mode AI Assistant</sub>", unsafe_allow_html=True)

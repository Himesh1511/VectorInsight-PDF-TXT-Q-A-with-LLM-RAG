import streamlit as st
import zipfile
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
body {
    background-color: #f7f8fa;
    margin: 0;
    padding: 0;
}

/* Reduce page bottom padding */
.block-container {
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* Main layout */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Sidebar styling */
.sidebar-content {
    padding: 1rem;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 200px);
}

/* Chat box styling */
.chat-box {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 1rem;
    background: #ffffff;
    border-radius: 12px;
    height: 100%;
    overflow-y: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 0 !important;
}

/* Messages */
.user-msg {
    align-self: flex-end;
    background: #DCF8C6;
    border-radius: 16px 16px 0 16px;
    padding: 0.6rem 0.9rem;
    max-width: 75%;
    word-wrap: break-word;
    margin-bottom: 0.2rem;
}

.bot-msg {
    align-self: flex-start;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px 16px 16px 0;
    padding: 0.6rem 0.9rem;
    max-width: 75%;
    word-wrap: break-word;
    margin-bottom: 0.2rem;
}

.typing {
    color: #6b7280;
    font-style: italic;
}

/* Input area */
.stTextInput {
    margin-top: 0.5rem !important;
}

/* Fix streamlit spacing */
.st-emotion-cache-1y4p8pa {
    padding-top: 0rem !important;
}

.st-emotion-cache-1wrcq25 {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 VectorInsight — AI Document Assistant")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "📂 Upload PDF/TXT/ZIP files",
        type=["pdf", "txt", "zip"],
        accept_multiple_files=True,
        help="Upload your documents here to analyze and chat with."
    )

    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.3, 0.05)
    k = st.slider("Top-K Chunks", 1, 10, 3)
    chunk_size = st.slider("Chunk Size", 200, 1200, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 80, 20)
    st.divider()
    st.button("🧹 Clear Chat", on_click=lambda: st.session_state.clear())

# ------------------ Helper Functions ------------------
def chunk_text(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "!", "?", ";", ","]
    )
    return splitter.split_text(text)

def extract_text_from_file(file, ext):
    if ext.lower() == ".pdf":
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        return text
    elif ext.lower() == ".txt":
        data = file.read()
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = str(data)
        return text
    return ""

def extract_text_from_zip(uploaded_zip):
    texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmpdir)
        for file in Path(tmpdir).rglob("*"):
            if file.suffix.lower() in [".pdf", ".txt"]:
                with open(file, "rb") as f:
                    text = extract_text_from_file(f, file.suffix)
                    texts.append(text)
    return "\n".join(texts)

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
if "index" not in st.session_state:
    st.session_state["index"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "text_data" not in st.session_state:
    st.session_state["text_data"] = ""

# ------------------ Process Upload ------------------
if uploaded_files:
    full_texts = []
    for file in uploaded_files:
        if file.name.endswith(".zip"):
            text = extract_text_from_zip(file)
        else:
            text = extract_text_from_file(file, Path(file.name).suffix)
        full_texts.append(text)

    combined_text = "\n".join(full_texts)
    if combined_text and combined_text != st.session_state["text_data"]:
        st.session_state["text_data"] = combined_text
        with st.spinner("Processing documents..."):
            chunks = chunk_text(combined_text, chunk_size, chunk_overlap)
            embeddings = embed_text(chunks, st.session_state["model"])
            index = build_faiss_index(embeddings)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
        # Auto-generate summary
        with st.spinner("Generating summary..."):
            summary_prompt = f"Summarise the following document clearly and concisely:\n\n{combined_text[:15000]}"
            summary = llama3_generate(summary_prompt, temperature)
            st.session_state["history"].append(("bot", f"📄 Document Summary:\n{summary}"))
        st.success("Documents processed and summary generated ✅")

# ------------------ Chat Interface ------------------
# Create a container for the entire chat section
chat_container = st.container()

with chat_container:
    # Add the chat container wrapper
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    st.markdown("### 💭 Chat Window")
    
    # Create the chat box
    chat_box = st.container()
    
    with chat_box:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for role, msg in st.session_state["history"]:
            css = "user-msg" if role == "user" else "bot-msg"
            st.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the chat container wrapper
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["history"].append(("user", user_input))
    with st.spinner("Thinking..."):
        if st.session_state["index"] is not None:
            dists, idxs = search_index(user_input, st.session_state["model"], st.session_state["index"], k)
            context = "\n\n".join([st.session_state["chunks"][i] for i in idxs])
            prompt = f"""You are a helpful assistant.
Answer the user's question using the context below.
If the context doesn't contain the answer, rely on general knowledge.

Context:
{context}

Question: {user_input}
Answer:"""
        else:
            prompt = user_input
        answer = llama3_generate(prompt, temperature)
        st.session_state["history"].append(("bot", answer))
    st.rerun()

st.markdown("<br><sub>🧠 VectorInsight Unified Document Chat Assistant</sub>", unsafe_allow_html=True)

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import io
import tempfile
from PyPDF2 import PdfReader
from groq import Groq

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(page_title="VectorInsight RAG", layout="wide")

# Initialize API client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f9fafc;
        padding: 1.5rem;
    }
    .st-emotion-cache-1v0mbdj {
        order: -1;
    }
    .chat-message {
        border-radius: 15px;
        padding: 12px 18px;
        margin: 8px 0;
        max-width: 80%;
        line-height: 1.6;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .user-msg {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .bot-msg {
        background-color: #f1f3f4;
    }
    .main-chat {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        height: 80vh;
        overflow-y: auto;
        padding: 20px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper functions
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    return embedder.encode(chunks)

def retrieve_relevant_chunks(query, chunks, embeddings, top_k=3):
    query_vec = embedder.encode([query])[0]
    scores = np.dot(embeddings, query_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_response(prompt, temperature=0.2):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an intelligent RAG-based assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content

def summarize_document(text):
    prompt = f"Summarize this document briefly:\n\n{text[:4000]}"
    return generate_response(prompt)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📁 Upload PDF/TXT/ZIP Files", type=["pdf", "txt", "zip"])

if st.sidebar.button("🧹 Clear Session"):
    st.session_state.clear()
    st.experimental_rerun()

temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.2)
top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 3)
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500)
overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 80)

# ----------------------------
# Chat initialization
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "summary_done" not in st.session_state:
    st.session_state.summary_done = False

# ----------------------------
# File Upload → Auto Summary
# ----------------------------
if uploaded_file and not st.session_state.summary_done:
    with st.spinner("Processing and summarizing document..."):
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        chunks = chunk_text(text, chunk_size, overlap)
        embeddings = embed_chunks(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

        summary = summarize_document(text)
        st.session_state.chat_history.append(("bot", f"📘 **Document Summary:** {summary}"))
        st.session_state.summary_done = True

# ----------------------------
# Chat Window
# ----------------------------
st.markdown("## 💬 Chat Window")
chat_container = st.container()

with chat_container:
    st.markdown('<div class="main-chat">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        role_class = "user-msg" if sender == "user" else "bot-msg"
        st.markdown(f'<div class="chat-message {role_class}">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# User Input
# ----------------------------
query = st.chat_input("Ask something about the document or chat with AI...")

if query:
    st.session_state.chat_history.append(("user", query))

    # RAG-based answer generation
    if st.session_state.chunks:
        relevant_chunks = retrieve_relevant_chunks(query, st.session_state.chunks, st.session_state.embeddings, top_k)
        context = "\n".join(relevant_chunks)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}"
        answer = generate_response(prompt, temperature)
    else:
        answer = generate_response(query, temperature)

    st.session_state.chat_history.append(("bot", answer))
    st.rerun()

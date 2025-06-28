import streamlit as st
import os
import zipfile
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import requests

# --- File Parsing Logic ---
def extract_text_from_file(file, filetype):
    if filetype == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    
    elif filetype == ".txt":
        return file.read().decode("utf-8", errors="ignore")
    
    return ""  # fallback

def extract_text_from_zip(uploaded_zip):
    extracted_texts = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(tmpdir)
        for file in Path(tmpdir).rglob("*"):
            if file.suffix in [".pdf", ".txt"]:
                with open(file, "rb") as f:
                    extracted_texts.append(extract_text_from_file(f, file.suffix))
    return "\n".join(extracted_texts)

# --- Text Chunking ---
def chunk_text(text, chunk_size=400, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ","]
    )
    return splitter.split_text(text)

# --- Load Embedding Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(query, model, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# --- LLaMA 3 Answer Generation ---
def generate_llama3_answer(query, context):
    api_key = st.secrets["GROQ_API_KEY"]
    url = "https://api.groq.com/openai/v1/chat/completions"

    prompt = f"""You are a helpful assistant. Use the context below to answer the question clearly.

Context:
{context}

Question: {query}
Answer:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"LLaMA 3 API Error: {response.text}")

# --- Streamlit UI ---
st.set_page_config(page_title="DocuQuery Multi-File + LLaMA 3", layout="centered")
st.title("📁 VectorInsight: Multi-File PDF/TXT/Q&A with LLaMA 3")

uploaded_files = st.file_uploader(
    "📤 Upload PDF, TXT, ZIP files (multiple supported)",
    type=["pdf", "txt", "zip"],
    accept_multiple_files=True
)
query = st.text_input("💬 Ask a question about the uploaded documents")

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        if file.name.endswith(".zip"):
            all_text += extract_text_from_zip(file) + "\n"
        elif file.name.endswith((".pdf", ".txt")):
            all_text += extract_text_from_file(file, Path(file.name).suffix) + "\n"
        else:
            st.warning(f"⚠️ Unsupported file type: {file.name}")

    if all_text.strip() == "":
        st.error("❌ No valid text extracted from the uploaded files.")
    else:
        with st.spinner("🔍 Processing documents..."):
            chunks = chunk_text(all_text)
            model = load_model()
            embeddings = embed_text(chunks, model)
            index = build_faiss_index(embeddings)

        st.success(f"✅ Processed {len(chunks)} chunks.")

        if query:
            with st.spinner("🧠 Searching and querying LLaMA 3..."):
                top_chunks = search_index(query, model, index, chunks)
                context = "\n\n".join(top_chunks)
                try:
                    answer = generate_llama3_answer(query, context)
                    st.subheader("🤖 LLaMA 3 Answer")
                    st.write(f"**Q:** {query}")
                    st.write(f"**A:** {answer}")
                except Exception as e:
                    st.error(str(e))

            st.subheader("📌 Top Matching Chunks")
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}:**\n> {chunk}")

else:
    st.info("📁 Upload at least one .pdf, .txt, or .zip file to get started.")

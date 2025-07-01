import streamlit as st
import os
import zipfile
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import requests
import pinecone

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

# --- Pinecone Functions ---
def init_pinecone(api_key, env, index_name, dim):
    pinecone.init(api_key=api_key, environment=env)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dim, metric="cosine")
    return pinecone.Index(index_name)

def upsert_embeddings(index, chunks, embeddings, namespace="default"):
    # Pinecone expects a list of (id, vector, metadata)
    vectors = [(str(i), emb.tolist(), {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
    # Pinecone's batch limit is 100 vectors per upsert
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size], namespace=namespace)

def query_pinecone(index, model, query, top_k=3, namespace="default"):
    query_vector = model.encode([query])[0].tolist()
    results = index.query(vector=query_vector, top_k=top_k, namespace=namespace, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

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
st.set_page_config(page_title="VectorInsight", layout="centered")
st.title("📁 VectorInsight: Multi-File PDF/TXT/Q&A with LLaMA 3")

uploaded_files = st.file_uploader(
    "📤 Upload PDF, TXT, ZIP files (multiple supported)",
    type=["pdf", "txt", "zip"],
    accept_multiple_files=True
)
query = st.text_input("💬 Ask a question about the uploaded documents")

# --- Pinecone Config ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX_NAME = "vectorinsight-index"
PINECONE_NAMESPACE = "default"  # can be customized/user-specific if needed

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
            
            # Init Pinecone and upsert embeddings
            pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, embeddings.shape[1])
            upsert_embeddings(pinecone_index, chunks, embeddings, namespace=PINECONE_NAMESPACE)

        st.success(f"✅ Processed {len(chunks)} chunks.")

        if query:
            with st.spinner("🧠 Searching and querying LLaMA 3..."):
                top_chunks = query_pinecone(pinecone_index, model, query, namespace=PINECONE_NAMESPACE)
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

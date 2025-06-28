import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import requests

# --- PDF Text Extraction ---
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# --- Improved Text Chunking ---
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

# --- Embed Chunks ---
def embed_text(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# --- Build Vector Index ---
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# --- Semantic Search ---
def search_index(query, model, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# --- Call LLaMA 3 API via Groq ---
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
st.set_page_config(page_title="DocuQuery + LLaMA 3", layout="centered")
st.title("📄 DocuQuery: PDF Q&A Powered by LLaMA 3")

uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])
query = st.text_input("💬 Ask a question about the document")

if uploaded_file:
    with st.spinner("🔍 Extracting and processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        model = load_model()
        embeddings = embed_text(chunks, model)
        index = build_faiss_index(embeddings)
    st.success("✅ Document processed!")

    if query:
        with st.spinner("🔎 Searching..."):
            results = search_index(query, model, index, chunks)
            context = "\n\n".join(results)

        st.subheader("🤖 Answer (via LLaMA 3)")
        try:
            answer = generate_llama3_answer(query, context)
            st.write(f"**Q:** {query}")
            st.write(f"**A:** {answer}")
        except Exception as e:
            st.error(f"❌ Failed to get answer from LLaMA 3: {e}")

        st.subheader("🔍 Top Matching Chunks")
        for i, chunk in enumerate(results, 1):
            st.markdown(f"**Chunk {i}:**\n> {chunk}")
else:
    st.info("📄 Upload a PDF to begin.")

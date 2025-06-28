import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# --- PDF Text Extraction ---
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# --- Text Chunking ---
def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# --- Load SentenceTransformer Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Generate Embeddings ---
def embed_text(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

# --- Build FAISS Index ---
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# --- Perform Similarity Search ---
def search_index(query, model, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# --- Simulate Answer without LLM ---
def naive_generate_answer(query, matched_chunks):
    if not matched_chunks:
        return "No relevant context found."
    best_chunk = matched_chunks[0]
    summary = best_chunk.split(".")[0] + "."  # Extract first sentence
    return f"Based on the document, here's what we found:\n\n{summary}"

# --- Streamlit UI ---
st.set_page_config(page_title="DocuQuery - PDF Search", layout="centered")
st.title("📄 DocuQuery: Semantic PDF Search (No LLM)")

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
        with st.spinner("🔎 Searching for relevant chunks..."):
            results = search_index(query, model, index, chunks)

            st.subheader("🔍 Top Matching Chunks")
            for i, chunk in enumerate(results, 1):
                st.markdown(f"**Chunk {i}:**\n> {chunk}")

            st.subheader("💡 Generated Answer (No LLM)")
            naive_answer = naive_generate_answer(query, results)
            st.write(f"**Q:** {query}")
            st.write(f"**A:** {naive_answer}")
else:
    st.info("📄 Please upload a PDF to begin.")

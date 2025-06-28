import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Functions

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

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

# Streamlit UI

st.set_page_config(page_title="RAG Mini App", layout="centered")
st.title("🔍 RAG Demo: Chunking + Vector Search")

uploaded_file = st.file_uploader("Upload a Research PDF", type=["pdf"])
query = st.text_input("Ask a question about the document")

if uploaded_file:
    with st.spinner("Extracting and processing..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        model = load_model()
        embeddings = embed_text(chunks, model)
        index = build_faiss_index(embeddings)
    st.success("Document processed!")

    if query:
        with st.spinner("Retrieving context..."):
            results = search_index(query, model, index, chunks)
            st.subheader("🔍 Top Matching Chunks")
            for i, chunk in enumerate(results, 1):
                st.markdown(f"**Chunk {i}:**\n> {chunk}")

            st.subheader("💡 Simulated Answer")
            st.write(f"Q: {query}")
            st.write("A: Based on the matched content, here's a possible answer contextually generated.")

else:
    st.info("Upload a PDF to get started.")


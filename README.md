# VectorInsight-A-RAG-Powered-PDF-Explorer



A simple Streamlit app that demonstrates Retrieval-Augmented Generation (RAG) using chunking, embedding, and vector search over PDFs. Upload a research PDF, ask questions about its content, and get contextually relevant answers based on semantic search.

## Features

- **PDF Upload**: Upload any research PDF for analysis.
- **Text Extraction**: Extracts text from all pages of the PDF.
- **Text Chunking**: Splits extracted text into overlapping chunks for better retrieval.
- **Semantic Embedding**: Uses `sentence-transformers` to embed each chunk.
- **FAISS Vector Search**: Builds a FAISS index for fast similarity search.
- **Question Answering**: Enter a question and retrieve the most relevant chunks from your document.
- **Streamlit UI**: Easy-to-use interface with upload, query, and result display.

## Demo

![demo-screenshot](demo-screenshot.png)  <!-- Add a screenshot if available -->

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)
- [sentence-transformers](https://www.sbert.net/)
- [langchain](https://python.langchain.com/)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- numpy

## Installation

```bash
# Create and activate a new environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install streamlit pymupdf sentence-transformers langchain faiss-cpu numpy
```

## Usage

```bash
streamlit run app.py
```

1. Open the web browser link provided by Streamlit.
2. Upload a PDF file.
3. Enter your question in the text input.
4. View the top-matching chunks and a simulated answer.

## How It Works

1. **PDF Parsing**: The app loads your PDF and extracts all text.
2. **Chunking**: The text is split into manageable, possibly overlapping chunks for better retrieval and context.
3. **Embedding**: Each chunk is embedded into a vector space using a pre-trained SentenceTransformer model.
4. **Indexing**: All chunk embeddings are indexed with FAISS for fast similarity search.
5. **Querying**: When you submit a question, it is embedded and searched against the index. The top-k most similar chunks are displayed.

## Code Overview

Key components of the app:

- `extract_text_from_pdf`: Extracts text from the entire PDF.
- `chunk_text`: Splits text into overlapping chunks.
- `load_model`: Loads the SentenceTransformer embedding model.
- `embed_text`: Embeds each chunk into a vector.
- `build_faiss_index`: Builds a FAISS flat index from the embeddings.
- `search_index`: Finds the most relevant text chunks for a given query.

## Limitations

- The "answer" is simulated; it displays the most relevant chunks but does not generate a new answer.
- Only supports research PDFs in English.
- For best results, use research papers with selectable (not scanned) text.

## License

MIT License

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [sentence-transformers](https://www.sbert.net/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)

---

*Built with ❤️ for rapid RAG prototyping!*

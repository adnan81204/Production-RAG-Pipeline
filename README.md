# 🚀 Production-Ready RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system built to solve the most common enterprise AI challenge: LLM hallucinations. This project implements a **Two-Stage Hybrid Retrieval Pipeline** using local vector storage, cross-encoder reranking, and ultra-low latency generation via the Groq API.

## 🧠 Architecture Overview

1. **Ingestion Engine (`ingest.py`):** Parses local PDFs, chunks text, and generates local embeddings (`all-MiniLM-L6-v2`) to a ChromaDB vector store.
2. **Advanced Retrieval (`app.py`):** Retrieves the top 10 relevant chunks via Vector search.
3. **Reranker:** Passes results through a FlashRank Cross-Encoder (`ms-marco-MiniLM`) to strictly grade and filter the chunks.
4. **Generation:** Injects the highest-scoring context into a prompt template and generates a factual response using Groq's `llama-3.1-8b-instant` model.

## ⚙️ Local Setup & Installation

**1. Clone the repository**
`git clone https://github.com/adnan81204/Production-RAG-Pipeline.git`

**2. Create a Virtual Environment**
`python -m venv venv`
`source venv/Scripts/activate`

**3. Install Dependencies**
`pip install langchain langchain-community langchain-chroma langchain-huggingface pypdf sentence-transformers rank_bm25 flashrank langchain-classic langchain-groq`

**4. Setup Environment Variables**
Add your free Groq API key in `app.py`:
`os.environ["GROQ_API_KEY"] = "YOUR_API_KEY_HERE"`

---
*Built by Shaik Adnan Tousef*

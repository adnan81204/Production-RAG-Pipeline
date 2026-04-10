import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# THE FIX: Import Ranker directly to control where the model downloads
from flashrank import Ranker

# 1. Load the "Memory" we created in Phase 1
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Setup the Basic Retriever (Top 10 results)
base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# 2. Setup the Reranker safely for Windows
print("--- Initializing Reranker (Downloading model to local folder) ---")
os.makedirs("./local_model_cache", exist_ok=True) # Creates a safe folder in your project

# Initialize the Ranker client explicitly with the safe folder
ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./local_model_cache")

# Pass the client to LangChain
compressor = FlashrankRerank(client=ranker_client)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# 3. Test the Advanced Search
query = "What are the challenges of Agentic AI?"
print(f"\n--- Searching for: {query} ---")

compressed_docs = compression_retriever.invoke(query)

print(f"\nFound {len(compressed_docs)} highly relevant chunks:")
for i, doc in enumerate(compressed_docs[:2]): 
    score = doc.metadata.get('relevance_score', 'N/A')
    print(f"\n[Result {i+1}] Score: {score}")
    print(f"Content: {doc.page_content[:200]}...")

print("\n✅ Phase 2 Complete: Advanced Retrieval System is ready.")
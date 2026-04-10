import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load Documents
print("--- Loading Documents ---")
loader = DirectoryLoader('./data', glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 2. Chunk Documents
print(f"--- Splitting {len(docs)} pages into chunks ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80
)
chunks = text_splitter.split_documents(docs)

# 3. Create Vector Store (This might take a minute the first time to download the model)
print(f"--- Creating Vector Store from {len(chunks)} chunks ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# 4. Simple Test
print("--- Testing Search ---")
query = "What is this document about?" 
results = vector_db.similarity_search(query, k=1)
if results:
    print("\nFound relevant text:\n", results[0].page_content[:300])

print("\n✅ Phase 1 Complete: Vector DB is saved in ./chroma_db")
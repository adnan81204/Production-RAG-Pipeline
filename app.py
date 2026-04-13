import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from flashrank import Ranker
from langchain_groq import ChatGroq

# 1. Setup API Key (Paste your Groq key here inside the quotes)
# ⚠️ WARNING: Never upload this key to GitHub!
os.environ["GROQ_API_KEY"] = "gsk_YOUR_API_KEY_HERE"

# 2. Load Memory & Retriever (From Phase 1 & 2)
print("--- Booting up AI Systems ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

ranker_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./local_model_cache")
compressor = FlashrankRerank(client=ranker_client)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# 3. Setup the LLM (The "Voice")
# We are using Llama 3 8B because it is lightning fast and very smart
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# 4. The Core RAG Logic
def ask_assistant(query):
    print(f"\n🧠 Searching memory for: '{query}'...")
    
    # Step A: Search the database using our Reranker
    docs = compression_retriever.invoke(query)
    
    if not docs:
        return "I couldn't find any relevant information in my documents."
        
    # Step B: Combine the best text chunks into one big string
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Step C: Give strict instructions to the LLM (Prompt Engineering)
    prompt = f"""You are a highly intelligent and factual assistant. 
    Answer the user's question using ONLY the provided context below. 
    If the answer is not contained in the context, you must reply: "I don't have enough information to answer that based on my documents."
    
    Context:
    {context}
    
    Question: {query}
    """
    
    # Step D: Get the answer
    print("💬 Generating answer...\n")
    response = llm.invoke(prompt)
    return response.content

# 5. Let's test the complete system!
if __name__ == "__main__":
    question = "What are the challenges of Agentic AI?"
    
    answer = ask_assistant(question)
    
    print("========================================")
    print("🤖 AI Answer:")
    print("========================================")
    print(answer)
    print("\n✅ Phase 3 Complete: End-to-End RAG System is alive!")

import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq  # <--- CHANGED: Using native client
from dotenv import load_dotenv

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "rag_engine", "faiss_index")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load API Key
load_dotenv(ENV_PATH)


# --- GLOBAL CACHE (Singleton Pattern) ---
# We keep these at the global level so they are only loaded ONCE.
# This is a 'Pro' optimization for Production environments like Railway.
CACHE = {
    "embeddings": None,
    "db": None,
    "client": None
}

def get_agile_advice(role, state, dominant_wave):
    """
    Optimized RAG Pipeline: Uses cached models to provide instant advice.
    """
    # 1. Initialize Cache if empty
    if CACHE["embeddings"] is None:
        print("🧠 Pro-Mode: Loading NLP Embeddings into RAM...")
        CACHE["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if CACHE["db"] is None:
        try:
            CACHE["db"] = FAISS.load_local(DB_PATH, CACHE["embeddings"], allow_dangerous_deserialization=True)
            print("✅ Vector DB loaded.")
        except:
            return "⚠️ Advice Error: Vector DB not initialized. Run build_vector_db.py."
            
    if CACHE["client"] is None:
        CACHE["client"] = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # 2. Search for Context
    query = f"Protocol for {state} state and {dominant_wave} waves for {role}"
    docs = CACHE["db"].similarity_search(query, k=2)
    context_text = "\n\n".join([d.page_content for d in docs])

    # 3. Call Groq
    try:
        prompt = f"""
        You are an Expert Agile Coach & Neuroscientist.
        CONTEXT: {context_text}
        SITUATION: Role: {role}, State: {state}, Wave: {dominant_wave}
        TASK: 1 actionable management recommendation (max 2 sentences). Cite rules if applicable.
        """

        chat_completion = CACHE["client"].chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"RAG Advisor Offline: {str(e)}"


# Test block
if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    print(get_agile_advice("DevOps", "Stressed", "High Beta"))

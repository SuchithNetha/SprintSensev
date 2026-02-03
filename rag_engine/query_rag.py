import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

# --- 1. CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SprintSense-RAG")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "rag_engine", "faiss_index")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

# --- 2. SINGLETON RESOURCE MANAGER ---
class RAGManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance.embeddings = None
            cls._instance.db = None
            cls._instance.client = None
        return cls._instance

    def initialize(self):
        """Loads heavy resources only when needed."""
        if not self.embeddings:
            logger.info("Loading NLP Embeddings Model...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not self.db:
            if not os.path.exists(DB_PATH):
                logger.error(f"Vector DB not found at {DB_PATH}")
                raise FileNotFoundError("FAISS index missing. Run build_vector_db.py.")
            self.db = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector Database linked.")

        if not self.client:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment.")
                raise ValueError("Missing API Credentials")
            self.client = Groq(api_key=api_key)

rag_manager = RAGManager()

def get_agile_advice(role, state, dominant_wave):
    """Production-grade RAG pipeline with error handling and resource management."""
    try:
        rag_manager.initialize()
        
        # Retrieval
        query = f"Management protocols for {state} cognitive status with {dominant_wave} activity in a {role} role."
        docs = rag_manager.db.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])

        # Inference
        prompt = f"""
        Role: {role}
        Cognitive State: {state}
        Biometric Signal: {dominant_wave}
        Agile Context: {context}
        
        Provide one highly specific, actionable Scrum-compliant recommendation (max 2 sentences).
        """

        response = rag_manager.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a Scrum Master with a PhD in Neuroscience."},
                      {"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"RAG Pipeline Failure: {e}")
        return "Standard Protocol: Prioritize task refinement and ensure team sync. (Advice module degraded)"

if __name__ == "__main__":
    print(get_agile_advice("Developer", "Focused", "Alpha"))

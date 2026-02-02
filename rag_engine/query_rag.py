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


def get_agile_advice(role, state, dominant_wave):
    """
    1. Search Vector DB for rules related to the State/Wave.
    2. Ask Groq for specific advice based on those rules.
    """

    # 1. Setup Retrieval (LangChain handles the DB search)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        # allow_dangerous_deserialization is needed for local files
        db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except RuntimeError:
        return "⚠️ Error: Vector DB not found. Run build_vector_db.py first."

    # 2. Search for Context
    query = f"Protocol for {state} state and {dominant_wave} waves for {role}"
    docs = db.similarity_search(query, k=2)
    context_text = "\n\n".join([d.page_content for d in docs])

    # 3. Call Groq LLM (Native Client)
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        prompt = f"""
        You are an Expert Agile Coach & Neuroscientist.

        CONTEXT from Scrum Guide:
        {context_text}

        SITUATION:
        - Role: {role}
        - Detected State: {state}
        - Biometric Signal: {dominant_wave}

        TASK:
        Provide 1 concise, actionable management recommendation (max 2 sentences).
        Cite the specific rule from the context if applicable.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",  # Or "mixtral-8x7b-32768"
            temperature=0.3,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error connecting to Groq: {str(e)}"


# Test block
if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    print(get_agile_advice("DevOps", "Stressed", "High Beta"))

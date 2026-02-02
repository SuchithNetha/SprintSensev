import os
import sys

# Add root to path to ensure imports work (Safety check)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "scrum_guide.txt")
DB_PATH = os.path.join(BASE_DIR, "rag_engine", "faiss_index")


def build_database():
    print(f"📚 Loading Scrum Guide from: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        print("Please ensure you created the 'data' folder and 'scrum_guide.txt' file.")
        return

    # 1. Load the Text
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    # 2. Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"🔹 Split into {len(texts)} chunks.")

    # 3. Create Embeddings (Downloads model if needed - ~100MB)
    print("🧠 Initializing Embedding Model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Build Vector Store
    print("⚡ Building FAISS Index...")
    db = FAISS.from_documents(texts, embeddings)

    # 5. Save Locally
    db.save_local(DB_PATH)
    print(f"✅ Success! Vector Database saved to {DB_PATH}")


if __name__ == "__main__":
    build_database()

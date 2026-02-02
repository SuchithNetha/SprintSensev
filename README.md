# 🧠 SprintSense: AI-Driven Cognitive Load Monitor

**SprintSense** is a full-stack "Neuro-Agile" management tool designed to quantify the intangible aspects of software development: **Developer Stress and Burnout.** Instead of relying on invasive hardware, SprintSense uses a **Multi-Layer Perceptron (MLP)** to generate synthetic biometric signals (EEG) based on work-environment proxies (Jira tickets, sleep quality, deadlines). A **RAG-based AI Agent** then provides real-time, Scrum-compliant management advice to optimize team health.

## 🚀 Key Features
* **Synthetic Biometric Engine:** An MLP Regressor that maps environmental factors (Workload, Sleep, Complexity) to realistic EEG brainwave patterns (Alpha, Beta, Delta, Theta).
* **Mental State Classification:** A Random Forest Classifier that detects 4 distinct cognitive states: *Focused, Stressed, Fatigued, Distracted*.
* **RAG Advisor (The "Brain"):** An LLM-powered agent (Groq/Llama3) that retrieves specific protocols from the *Scrum Guide* to suggest interventions for stressed developers.
* **Real-Time Dashboard:** A Streamlit interface for Managers to monitor team health and visualize live cognitive load.

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **Backend API:** FastAPI
* **Machine Learning:** Scikit-Learn (MLP, Random Forest)
* **GenAI / RAG:** LangChain, FAISS (Vector DB), Groq API
* **Data Processing:** Pandas, NumPy

## 📂 Project Structure
```text
SprintSense/
├── api/                # FastAPI Backend
│   └── main.py         # API Endpoints & Logic
├── frontend/           # Streamlit Dashboard
│   └── app.py          # UI Logic & Visualization
├── ml_engine/          # Machine Learning Core
│   ├── data_gen.py     # Synthetic Data Generator
│   ├── train.py        # Model Training Script
│   └── artifacts/      # Saved Models (.pkl)
├── rag_engine/         # Retrieval Augmented Generation
│   ├── build_db.py     # Vector Database Builder
│   └── query_rag.py    # LLM Context Logic
├── tests/              # Automated Test Suite
└── data/               # Knowledge Base (Scrum Guide)

## 🚀 Deployment & Local Setup

### 1. Environment Variables
Create a `.env` file in the root directory (see `.env.example`):
```text
GROQ_API_KEY=your_key_here
```

### 2. Local Setup (Manual)
1. **Start Backend**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start Frontend**:
   ```bash
   streamlit run frontend/app.py
   ```

### 3. Docker Deployment (Recommended)
The project includes a `docker-compose.yml` for easy multi-container setup.
```bash
docker-compose up --build
```
* **API**: http://localhost:8000
* **Dashboard**: http://localhost:8501

### 4. Production Note
For production deployments (e.g., Render, Heroku), ensure the `API_URL` environment variable is set in the frontend service to point to your live API endpoint.

## 💜 Heroku Deployment

Heroku works best when you deploy the **API** and **Frontend** as two separate apps.

### App 1: The API (FastAPI)
1. Create a new app: `sprintsense-api`
2. Set Environment Variables: `GROQ_API_KEY`
3. Deploy the code. The `Procfile` will automatically run:
   `web: uvicorn api.main:app --host 0.0.0.0 --port $PORT`

### App 2: The Dashboard (Streamlit)
1. Create a new app: `sprintsense-ui`
2. Set Environment Variables: 
   * `API_URL`: `https://sprintsense-api.herokuapp.com/predict`
   * `GROQ_API_KEY`: (Needed for initialization)
3. Deploy the code. 
4. **CRITICAL**: Go to the Heroku dashboard and change the "Resources" command for the web dyno to:
   `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0`

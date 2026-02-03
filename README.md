# 🧠 SprintSense: AI-Driven Cognitive Load Monitor

**SprintSense** is a full-stack "Neuro-Agile" management tool designed to quantify the intangible aspects of software development: **Developer Stress and Burnout.** It uses a **Multi-Layer Perceptron (MLP)** to generate synthetic biometric signals (EEG) based on work-environment proxies, while a **RAG-based AI Agent** provides real-time, Scrum-compliant management advice.

## 🔗 Live Deployment
* **📊 Live Dashboard:** [https://streamlit-production-d362.up.railway.app/](https://streamlit-production-d362.up.railway.app/)
* **🧠 Neural Engine (API):** [https://web-production-6c04b.up.railway.app](https://web-production-6c04b.up.railway.app)

## 🚀 Key Features
* **Synthetic Biometric Engine:** Maps workload and environment factors to realistic EEG patterns (Alpha, Beta, Delta, Theta).
* **Mental State Classification:** Detects distinct cognitive states: *Focused, Stressed, Fatigued, Distracted*.
* **RAG Advisor:** An LLM-powered agent (Groq/Llama3) that suggests interventions based on the *Scrum Guide*.
* **Executive Dashboard:** Live visualization of team cognitive load and health trends.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI
* **ML:** Scikit-Learn (MLP, Random Forest)
* **GenAI:** LangChain, FAISS, Groq API

## � Local Setup (Docker)
The easiest way to run SprintSense locally is using Docker Compose.

1. **Environment Variables**: Create a `.env` file with your `GROQ_API_KEY`.
2. **Build & Run**:
   ```bash
   docker-compose up --build
   ```
* **API**: http://localhost:8000
* **Dashboard**: http://localhost:8501

---
*Deployed on Railway*

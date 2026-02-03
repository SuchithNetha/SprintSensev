import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .schemas import AssessmentRequest, PredictionResponse
import joblib
import numpy as np
import os
import uvicorn
import sys

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SprintSense-API")

# --- 2. SETUP & IMPORTS ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.db_manager import log_prediction, get_history
    from rag_engine.query_rag import get_agile_advice
    logger.info("Successfully imported internal modules.")
except ImportError as e:
    logger.error(f"Module import failed: {e}")

app = FastAPI(
    title="🧠 SprintSense AI API",
    version="2.1",
    description="Production-ready Cognitive Load & Management Advisor"
)

# --- 3. MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Path: {request.url.path} | Time: {process_time:.4f}s")
    return response

# --- 4. RESOURCE LOADING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_engine", "artifacts")

mlp_model = None
rf_model = None

@app.on_event("startup")
async def load_resources():
    global mlp_model, rf_model
    try:
        mlp_model = joblib.load(os.path.join(ARTIFACTS_DIR, "mlp_eeg_generator.pkl"))
        rf_model = joblib.load(os.path.join(ARTIFACTS_DIR, "rf_state_classifier.pkl"))
        logger.info("✅ ML Models loaded into memory.")
    except Exception as e:
        logger.error(f"❌ ML Loading Error: {e}")

# --- 5. ENDPOINTS ---

@app.get("/health")
@app.get("/")
def health_check():
    return {
        "status": "active",
        "engine": "SprintSense 2.1",
        "deployment": os.getenv("RAILWAY_ENVIRONMENT", "production"),
        "models_loaded": mlp_model is not None
    }

@app.get("/history")
def fetch_history():
    """Returns historical logs for trend analysis."""
    try:
        return get_history()
    except Exception as e:
        logger.error(f"History Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Database Error")

@app.post("/predict", response_model=PredictionResponse)
def predict_cognitive_state(data: AssessmentRequest):
    """
    Unified Pipeline: ML Generation -> RAG Advice -> DB Logging
    """
    if mlp_model is None or rf_model is None:
        raise HTTPException(status_code=503, detail="ML Models not initialized")

    try:
        # --- PART A: ML PREDICTION ---
        input_vector = np.array([[
            data.ticket_volume,
            data.deadline_proximity,
            data.sleep_quality,
            data.complexity,
            data.interruptions
        ]])

        synthetic_eeg_raw = mlp_model.predict(input_vector)
        eeg_values = synthetic_eeg_raw[0]
        state_prediction = rf_model.predict(synthetic_eeg_raw)[0]

        # --- PART B: RAG ADVICE ---
        wave_names = ["Alpha", "Beta", "Delta", "Theta"]
        dominant_idx = np.argmax(eeg_values)
        dominant_wave = wave_names[dominant_idx]

        try:
            ai_advice_text = get_agile_advice(data.role, state_prediction, dominant_wave)
        except Exception as e:
            logger.warning(f"RAG Error: {e}")
            ai_advice_text = "Analysis unavailable. Context sync error."

        # --- PART C: DB LOGGING ---
        eeg_dict = {
            "alpha": float(eeg_values[0]),
            "beta": float(eeg_values[1]),
            "delta": float(eeg_values[2]),
            "theta": float(eeg_values[3])
        }
        
        try:
            log_prediction(data.name, data.role, state_prediction, eeg_dict)
        except Exception as e:
            logger.error(f"DB Logging Failed: {e}")

        return {
            "state": state_prediction,
            "eeg_data": eeg_dict,
            "advice": ai_advice_text
        }

    except Exception as e:
        logger.error(f"Prediction Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Processing Error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

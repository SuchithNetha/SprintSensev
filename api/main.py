from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn
import sys

# --- 1. SETUP & IMPORTS ---
# We add the root folder to sys.path so we can import 'rag_engine'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database manager
try:
    from database.db_manager import log_prediction, get_history
    from rag_engine.query_rag import get_agile_advice
except ImportError:
    print("⚠️ Warning: Module import failed.")

app = FastAPI(title="🧠 SprintSense AI API", version="2.0", description="Cognitive Load & Management Advisor")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_engine", "artifacts")

# Load Models
try:
    mlp_model = joblib.load(os.path.join(ARTIFACTS_DIR, "mlp_eeg_generator.pkl"))
    rf_model = joblib.load(os.path.join(ARTIFACTS_DIR, "rf_state_classifier.pkl"))
    print("✅ ML Models loaded into memory.")
except Exception as e:
    print(f"❌ ML Loading Error: {e}")

# --- 2. DATA SCHEMAS ---
class AssessmentRequest(BaseModel):
    name: str  # [PRO] Added name for logging
    role: str
    ticket_volume: int 
    deadline_proximity: int 
    sleep_quality: int 
    complexity: int 
    interruptions: int 


class PredictionResponse(BaseModel):
    state: str
    eeg_data: dict
    advice: str


# --- 3. ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "engine": "SprintSense 2.0", "deployment": "Railway"}


@app.get("/history")
def fetch_history():
    """[PRO] Returns historical logs for trend analysis."""
    return get_history()


@app.post("/predict", response_model=PredictionResponse)
def predict_cognitive_state(data: AssessmentRequest):
    """
    Unified Pipeline: ML Generation -> RAG Advice -> DB Logging
    """

    # --- PART A: ML PREDICTION ---
    input_vector = np.array([[
        data.ticket_volume,
        data.deadline_proximity,
        data.sleep_quality,
        data.complexity,
        data.interruptions
    ]])

    # 1. Generate EEG & State
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
        ai_advice_text = "Analysis unavailable. Context sync error."
        print(f"RAG Error: {e}")

    # --- PART C: DB LOGGING [NEW PRO FEATURE] ---
    eeg_dict = {
        "alpha": float(eeg_values[0]),
        "beta": float(eeg_values[1]),
        "delta": float(eeg_values[2]),
        "theta": float(eeg_values[3])
    }
    
    try:
        log_prediction(data.name, data.role, state_prediction, eeg_dict)
    except Exception as e:
        print(f"DB Logging Failed: {e}")

    return {
        "state": state_prediction,
        "eeg_data": eeg_dict,
        "advice": ai_advice_text
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

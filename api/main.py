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

# Import the RAG function we just wrote
try:
    from rag_engine.query_rag import get_agile_advice
except ImportError:
    print("⚠️ Warning: Could not import rag_engine. Make sure query_rag.py exists.")


    # Fallback function if import fails
    def get_agile_advice(role, state, wave):
        return "AI Advice Unavailable"

app = FastAPI(title="SprintSense API", version="2.0")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "ml_engine", "artifacts")

print(f"🔄 Loading models from {ARTIFACTS_DIR}...")

try:
    mlp_model = joblib.load(os.path.join(ARTIFACTS_DIR, "mlp_eeg_generator.pkl"))
    rf_model = joblib.load(os.path.join(ARTIFACTS_DIR, "rf_state_classifier.pkl"))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load models. {e}")


# --- 2. DATA SCHEMAS ---
class AssessmentRequest(BaseModel):
    role: str  # [NEW] Needed for RAG Context (e.g., "DevOps")
    ticket_volume: int  # 0-3
    deadline_proximity: int  # 0-3
    sleep_quality: int  # 0-3
    complexity: int  # 0-3
    interruptions: int  # 0-3


class PredictionResponse(BaseModel):
    state: str
    eeg_data: dict
    advice: str  # [NEW] The AI Advice text


# --- 3. ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "system": "SprintSense API v2"}


@app.post("/predict", response_model=PredictionResponse)
def predict_cognitive_state(data: AssessmentRequest):
    """
    Full Pipeline:
    1. ML: Survey -> Synthetic EEG -> Mental State
    2. RAG: State + Context -> Groq -> Advice
    """

    # --- PART A: ML PREDICTION ---
    input_vector = np.array([[
        data.ticket_volume,
        data.deadline_proximity,
        data.sleep_quality,
        data.complexity,
        data.interruptions
    ]])

    # 1. Generate EEG
    synthetic_eeg_raw = mlp_model.predict(input_vector)
    eeg_values = synthetic_eeg_raw[0]  # Flatten

    # 2. Predict State
    state_prediction = rf_model.predict(synthetic_eeg_raw)[0]

    # --- PART B: RAG ADVICE GENERATION ---

    # Determine Dominant Wave (Highest value) for the prompt
    # Index 0=Alpha, 1=Beta, 2=Delta, 3=Theta (matches training order)
    wave_names = ["Alpha", "Beta", "Delta", "Theta"]
    dominant_idx = np.argmax(eeg_values)
    dominant_wave = wave_names[dominant_idx]

    # Call the RAG Engine
    # We wrap this in try/except so the API doesn't crash if Groq is down
    try:
        ai_advice_text = get_agile_advice(data.role, state_prediction, dominant_wave)
    except Exception as e:
        ai_advice_text = "Analysis unavailable. Please consult Scrum Master."
        print(f"RAG Error: {e}")

    # Return everything
    return {
        "state": state_prediction,
        "eeg_data": {
            "alpha": float(eeg_values[0]),
            "beta": float(eeg_values[1]),
            "delta": float(eeg_values[2]),
            "theta": float(eeg_values[3])
        },
        "advice": ai_advice_text
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

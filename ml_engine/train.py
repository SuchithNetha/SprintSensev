import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# --- 1. DYNAMIC PATH CONFIGURATION ---
# Get the folder where THIS script (train.py) is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script's location
# This works regardless of whether you run it from root or inside ml_engine
DATA_PATH = os.path.join(SCRIPT_DIR, "datasets", "synthetic_eeg_data.csv")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_models():
    # Debug Print to show you exactly where it is looking
    print(f"🔍 Looking for data at: {DATA_PATH}")

    # Verify file exists first
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found!")
        print(f"   Current Working Directory: {os.getcwd()}")
        return

    print(f"✅ Found data! Loading...")
    df = pd.read_csv(DATA_PATH)

    # --- 1. TRAIN MODEL A: MLP Regressor (Survey -> EEG) ---
    print("\n--- Training Model A: MLP (Questionnaire -> EEG) ---")

    X_survey = df[['ticket_volume', 'deadline_proximity', 'sleep_quality', 'complexity', 'interruptions']]
    y_eeg = df[['eeg_alpha', 'eeg_beta', 'eeg_delta', 'eeg_theta']]

    X_train, X_test, y_train, y_test = train_test_split(X_survey, y_eeg, test_size=0.2, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    preds = mlp.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"✅ MLP Training Complete. Mean Squared Error: {mse:.4f}")

    # Save using the absolute path
    joblib.dump(mlp, os.path.join(ARTIFACTS_DIR, "mlp_eeg_generator.pkl"))

    # --- 2. TRAIN MODEL B: Random Forest (EEG -> State) ---
    print("\n--- Training Model B: Random Forest (EEG -> Mental State) ---")

    X_eeg = df[['eeg_alpha', 'eeg_beta', 'eeg_delta', 'eeg_theta']]
    y_state = df['state_label']

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_eeg, y_state, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_rf, y_train_rf)

    preds_rf = rf.predict(X_test_rf)
    acc = accuracy_score(y_test_rf, preds_rf)
    print(f"✅ Random Forest Training Complete. Accuracy: {acc * 100:.2f}%")

    joblib.dump(rf, os.path.join(ARTIFACTS_DIR, "rf_state_classifier.pkl"))

    print(f"\n🎉 All models saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    train_models()

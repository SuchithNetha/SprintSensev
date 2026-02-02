import pandas as pd
import numpy as np
import os

# CONFIG
NUM_SAMPLES = 2000
# Ensure directory exists
os.makedirs("datasets", exist_ok=True)
OUTPUT_PATH = "datasets/synthetic_eeg_data.csv"


def generate_synthetic_data():
    print(f"Generating {NUM_SAMPLES} synthetic samples...")

    # 1. Generate Input Features (The Questionnaire Answers 0-3)
    # 0 = Good/Low Intensity, 3 = Bad/High Intensity
    data = {
        'ticket_volume': np.random.choice([0, 1, 2, 3], NUM_SAMPLES, p=[0.2, 0.4, 0.3, 0.1]),
        'deadline_proximity': np.random.choice([0, 1, 2, 3], NUM_SAMPLES, p=[0.3, 0.3, 0.2, 0.2]),
        'sleep_quality': np.random.choice([0, 1, 2, 3], NUM_SAMPLES, p=[0.1, 0.5, 0.3, 0.1]),  # 3 = Poor Sleep
        'complexity': np.random.choice([0, 1, 2, 3], NUM_SAMPLES),
        'interruptions': np.random.choice([0, 1, 2, 3], NUM_SAMPLES),
    }

    df = pd.DataFrame(data)

    # 2. Generate Target EEG Variables (The "Synthetic Brain")
    # We use linear combinations + noise to create "biological" patterns.

    # --- BETA WAVE (Stress/Focus) ---
    # Driven by Deadlines and Complexity.
    # Formula: High Load = High Beta.
    stress_factor = (df['ticket_volume'] * 0.4 + df['deadline_proximity'] * 0.4 + df['complexity'] * 0.2)
    # Normalize to 0.0 - 1.0 range with some randomness
    df['eeg_beta'] = (stress_factor / 3.0) + np.random.normal(0, 0.05, NUM_SAMPLES)
    df['eeg_beta'] = df['eeg_beta'].clip(0, 1)

    # --- ALPHA WAVE (Relaxation) ---
    # Inverse of Beta. High Stress = Low Alpha.
    df['eeg_alpha'] = 1 - df['eeg_beta'] + np.random.normal(0, 0.05, NUM_SAMPLES)
    df['eeg_alpha'] = df['eeg_alpha'].clip(0, 1)

    # --- DELTA WAVE (Fatigue) ---
    # Strongly driven by Sleep Quality (Q3).
    fatigue_factor = df['sleep_quality']
    df['eeg_delta'] = (fatigue_factor / 3.0) + np.random.normal(0, 0.05, NUM_SAMPLES)
    df['eeg_delta'] = df['eeg_delta'].clip(0, 1)

    # --- THETA WAVE (Distraction) ---
    # Driven by Interruptions.
    distraction_factor = df['interruptions']
    df['eeg_theta'] = (distraction_factor / 3.0) + np.random.normal(0, 0.05, NUM_SAMPLES)
    df['eeg_theta'] = df['eeg_theta'].clip(0, 1)

    # 3. Generate The "State Label" (Ground Truth for Random Forest)
    # These rules define what constitutes each state.
    conditions = [
        (df['eeg_delta'] > 0.65),  # Rule 1: High Delta = Fatigued
        (df['eeg_beta'] > 0.70),  # Rule 2: High Beta = Stressed
        (df['eeg_theta'] > 0.65),  # Rule 3: High Theta = Distracted
        (df['eeg_beta'] > 0.4) & (df['eeg_beta'] <= 0.70)  # Rule 4: Mid Beta = Focused
    ]
    choices = ['Fatigued', 'Stressed', 'Distracted', 'Focused']

    # Default is 'Relaxed' if none match
    df['state_label'] = np.select(conditions, choices, default='Relaxed')

    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Data saved to {OUTPUT_PATH}")
    print("Preview:")
    print(df[['ticket_volume', 'eeg_beta', 'state_label']].head())


if __name__ == "__main__":
    generate_synthetic_data()

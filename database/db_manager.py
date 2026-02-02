import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "sprintsense.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stress_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            name TEXT,
            role TEXT,
            state TEXT,
            alpha REAL,
            beta REAL,
            delta REAL,
            theta REAL
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(name, role, state, eeg_data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stress_logs (timestamp, name, role, state, alpha, beta, delta, theta)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name,
        role,
        state,
        eeg_data['alpha'],
        eeg_data['beta'],
        eeg_data['delta'],
        eeg_data['theta']
    ))
    conn.commit()
    conn.close()

def get_history():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM stress_logs ORDER BY timestamp DESC LIMIT 50')
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize on import
init_db()

import sqlite3
import os
import logging
from datetime import datetime
from contextlib import contextmanager

# --- 1. CONFIG & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SprintSense-DB")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "sprintsense.db")

# --- 2. CONNECTION POOLING (MIMIC) ---
@contextmanager
def get_db_connection():
    """Context manager for database connections ensuring safety."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row # Return results as dict-like
    try:
        yield conn
    finally:
        conn.close()

# --- 3. DATABASE OPERATIONS ---

def init_db():
    """Initializes the database schema with performance indexes."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stress_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    name TEXT NOT NULL,
                    role TEXT,
                    state TEXT NOT NULL,
                    alpha REAL,
                    beta REAL,
                    delta REAL,
                    theta REAL
                )
            ''')
            # Create index on common lookup field
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON stress_logs(timestamp)')
            conn.commit()
            logger.info("✅ Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def log_prediction(name, role, state, eeg_data):
    """Logs a prediction result safely."""
    query = '''
        INSERT INTO stress_logs (name, role, state, alpha, beta, delta, theta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    try:
        with get_db_connection() as conn:
            conn.execute(query, (
                name, role, state,
                eeg_data.get('alpha'),
                eeg_data.get('beta'),
                eeg_data.get('delta'),
                eeg_data.get('theta')
            ))
            conn.commit()
            logger.info(f"Logged state '{state}' for {name}.")
    except Exception as e:
        logger.error(f"Logging Failed: {e}")

def get_history(limit=50):
    """Fetches prediction history for trends."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM stress_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
            # Convert rows to list of tuples for API compatibility
            return [list(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"History Fetch Failed: {e}")
        return []

# Initialize on import
init_db()

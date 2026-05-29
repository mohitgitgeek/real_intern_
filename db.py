"""SQLite storage for analysis history (standard-library sqlite3, no extra deps)."""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intern_history.db')


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                snippet TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_factors TEXT
            )
            """
        )


def save_analysis(text, prediction, confidence, risk_factors):
    snippet = (text or '')[:300]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO analyses (created_at, snippet, prediction, confidence, risk_factors) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), snippet, prediction,
             float(confidence), '; '.join(risk_factors or [])),
        )


def get_history(limit=50):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at, snippet, prediction, confidence, risk_factors "
            "FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM analyses")


def stats():
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM analyses").fetchone()['c']
        fraud = conn.execute(
            "SELECT COUNT(*) AS c FROM analyses WHERE prediction = 'Fraudulent'"
        ).fetchone()['c']
    return {'total': total, 'fraud': fraud, 'legit': total - fraud}

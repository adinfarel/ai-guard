"""
src/ai_guard/storage/database.py

SQLite storage utilities for AI-Guard.

This module stores prediction logs for audit, monitoring, and future drift checks.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from numpy import isin

DEFAULT_DB_PATH = Path("logs/ai_guard.db")

def get_utc_timestamp() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """
    Initialize SQLite database and create prediction_logs table.
    """
    if not isinstance(db_path, Path):
        db_path = Path(db_path)
        
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    prompt TEXT,
                    decision TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    blocked_by TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    nlp_score REAL,
                    latency_ms REAL,
                    raw_response TEXT NOT NULL
            )
            """
        )
        
        conn.commit()
    
def insert_prediction_log(
    *,
    endpoint: str,
    prompt: str,
    decision: str,
    allowed: bool,
    blocked_by: list[str],
    reason: str,
    nlp_score: float | None,
    latency_ms: float | None,
    raw_response: dict[str, Any],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """
    Insert one prediction log row into SQLite.
    """
    if not isinstance(db_path, Path):
        db_path = Path(db_path)
    
    init_db(db_path)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                timestamp,
                endpoint,
                prompt,
                decision,
                allowed,
                blocked_by,
                reason,
                nlp_score,
                latency_ms,
                raw_response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                get_utc_timestamp(),
                endpoint,
                prompt,
                decision,
                int(allowed),
                json.dumps(blocked_by),
                reason,
                nlp_score,
                latency_ms,
                json.dumps(raw_response),
            ),
        )
        
        conn.commit()

def fetch_recent_logs(
    limit: int = 20,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch recent prediction logs."""
    if not isinstance(db_path, Path):
        db_path = Path(db_path)
    
    if not db_path.exists():
        return []
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT *
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit, ),
        )
        
        rows = cursor.fetchall()
    
    return [dict(row) for row in rows]
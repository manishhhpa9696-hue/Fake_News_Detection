"""
SQLite Database Module for Analysis History.
Stores privacy-preserving audit logs of predictions, calibrated confidence,
timestamps, and linguistic markers, with full user controls to search, filter, and purge.
"""

import os
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "news_history.db"
)


def get_db_connection(db_path: str = DB_FILE_PATH) -> sqlite3.Connection:
    """Creates a connection to SQLite with row-dictionary access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_FILE_PATH) -> None:
    """Initializes the database schema if not already present."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS article_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            headline TEXT,
            text_snippet TEXT,
            text_hash TEXT NOT NULL,
            url TEXT,
            domain TEXT,
            language TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            uncertainty_flag INTEGER NOT NULL,
            clickbait_score REAL,
            temporal_score REAL,
            geo_score REAL,
            model_name TEXT NOT NULL
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON article_history(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction ON article_history(prediction)")

    # Migration checks for existing databases
    cursor.execute("PRAGMA table_info(article_history)")
    cols = [info[1] for info in cursor.fetchall()]
    if "temporal_score" not in cols:
        try:
            cursor.execute("ALTER TABLE article_history ADD COLUMN temporal_score REAL")
        except Exception:
            pass
    if "geo_score" not in cols:
        try:
            cursor.execute("ALTER TABLE article_history ADD COLUMN geo_score REAL")
        except Exception:
            pass

    conn.commit()
    conn.close()


def save_prediction_record(
    headline: str,
    text: str,
    url: Optional[str],
    domain: Optional[str],
    language: str,
    prediction: str,
    confidence: float,
    uncertainty_flag: bool,
    clickbait_score: float,
    model_name: str,
    temporal_score: Optional[float] = None,
    geo_score: Optional[float] = None,
    db_path: str = DB_FILE_PATH,
) -> int:
    """
    Inserts a privacy-preserving analysis record.
    Full text is hashed with SHA-256; only the first 200 characters are stored for audit.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    text_clean = (text or "").strip()
    text_hash = hashlib.sha256(text_clean.encode("utf-8")).hexdigest()
    text_snippet = text_clean[:200] + ("..." if len(text_clean) > 200 else "")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO article_history (
            timestamp, headline, text_snippet, text_hash, url, domain,
            language, prediction, confidence, uncertainty_flag,
            clickbait_score, temporal_score, geo_score, model_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp_str,
            (headline or "")[:300],
            text_snippet,
            text_hash,
            (url or "")[:500],
            (domain or "")[:100],
            language,
            prediction,
            round(confidence, 2),
            1 if uncertainty_flag else 0,
            round(clickbait_score, 1) if clickbait_score is not None else None,
            round(temporal_score, 1) if temporal_score is not None else None,
            round(geo_score, 1) if geo_score is not None else None,
            model_name,
        ),
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id


def get_history_records(
    limit: int = 100,
    search_query: Optional[str] = None,
    prediction_filter: Optional[str] = None,
    db_path: str = DB_FILE_PATH,
) -> List[Dict[str, Any]]:
    """Retrieves analysis records with optional filtering."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM article_history WHERE 1=1"
    params = []

    if prediction_filter and prediction_filter != "All":
        query += " AND prediction = ?"
        params.append(prediction_filter)

    if search_query and search_query.strip():
        query += " AND (headline LIKE ? OR text_snippet LIKE ? OR domain LIKE ?)"
        term = f"%{search_query.strip()}%"
        params.extend([term, term, term])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_record_by_id(record_id: int, db_path: str = DB_FILE_PATH) -> bool:
    """Deletes a specific record by primary key."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM article_history WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_history(db_path: str = DB_FILE_PATH) -> int:
    """Clears all saved audit logs."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM article_history")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def get_history_summary_stats(db_path: str = DB_FILE_PATH) -> Dict[str, Any]:
    """Computes summary statistics of saved predictions."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM article_history")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM article_history WHERE prediction LIKE '%Fake%'")
    fake_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM article_history WHERE prediction LIKE '%Reliable%'")
    reliable_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM article_history WHERE uncertainty_flag = 1")
    uncertain_count = cursor.fetchone()[0]

    conn.close()
    return {
        "total_analyzed": total,
        "fake_count": fake_count,
        "reliable_count": reliable_count,
        "uncertain_count": uncertain_count,
    }

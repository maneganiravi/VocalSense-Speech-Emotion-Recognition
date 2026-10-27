import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "vocalsense.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                emotion TEXT NOT NULL,
                cnn_emotion TEXT NOT NULL,
                cnn_bilstm_emotion TEXT NOT NULL,
                probabilities_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                confidence REAL,
                model_used TEXT NOT NULL DEFAULT 'ensemble'
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "confidence" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
        if "model_used" not in columns:
            connection.execute(
                "ALTER TABLE predictions ADD COLUMN model_used TEXT NOT NULL DEFAULT 'ensemble'"
            )


def save_prediction(
    filename: str,
    duration_seconds: float,
    emotion: str,
    cnn_emotion: str,
    cnn_bilstm_emotion: str,
    probabilities: Dict[str, Any],
    confidence: Optional[float] = None,
    model_used: str = "ensemble",
) -> Dict[str, Any]:
    if confidence is None:
        ensemble = probabilities.get("ensemble", {})
        confidence = ensemble.get("confidence")
        if confidence is None:
            class_probabilities = ensemble.get("probabilities", {})
            if class_probabilities:
                confidence = max(class_probabilities.values())
    if (
        not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("A valid ensemble confidence is required to save a prediction.")

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                filename, duration_seconds, emotion, cnn_emotion,
                cnn_bilstm_emotion, probabilities_json, confidence, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                duration_seconds,
                emotion,
                cnn_emotion,
                cnn_bilstm_emotion,
                json.dumps(probabilities),
                float(confidence),
                model_used,
            ),
        )
        row = connection.execute(
            "SELECT id, filename, duration_seconds, emotion, confidence, model_used, created_at "
            "FROM predictions WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    if row is None:
        raise RuntimeError("The saved prediction could not be read back.")
    return dict(row)


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, filename, duration_seconds, emotion, confidence, model_used, created_at "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_prediction_count() -> int:
    with _connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM predictions").fetchone()
    if row is None:
        raise RuntimeError("The prediction count could not be read from the database.")
    return int(row["count"])


def get_prediction_details(prediction_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, filename, duration_seconds, emotion, cnn_emotion, model_used,
                   cnn_bilstm_emotion, probabilities_json, confidence, created_at
            FROM predictions WHERE id = ?
            """,
            (prediction_id,),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    try:
        result["probabilities"] = json.loads(result.pop("probabilities_json"))
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Prediction {prediction_id} has invalid stored probability data."
        ) from error
    return result
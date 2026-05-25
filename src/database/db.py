from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "medseg.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    from src.database.models import ALL_TABLES
    with get_conn() as conn:
        for stmt in ALL_TABLES:
            conn.execute(stmt)
        conn.commit()

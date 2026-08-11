import sqlite3
from pathlib import Path

DB_PATH = Path("data/database/nba.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

print(f"Created database: {DB_PATH}")

conn.close()
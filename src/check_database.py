import sqlite3

db_path = "data/database/nba.db"

conn = sqlite3.connect(db_path)

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()

print("Tables in database:")

for table in tables:
    print(f"  OK {table[0]}")

conn.close()
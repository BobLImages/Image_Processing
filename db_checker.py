#db_checker.py

import sqlite3
import pandas as pd  # optional, for nice table list
from pathlib import Path

DB_PATH = Path("D:\\Image Data Files SQL\\2024-09-07 Eastern View vs Hugoenot Football.db")  # ← CHANGE THIS to your actual .db file path

def inspect_database(db_path):
    print(f"🔍 Opening database: {db_path}\n")
    
    conn = sqlite3.connect(db_path)
    
    # 1. List all tables
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = conn.execute(tables_query).fetchall()
    
    if not tables:
        print("No tables found in the database.")
        conn.close()
        return
    
    print(f"Found {len(tables)} table(s):\n")
    for (table_name,) in tables:
        print(f"📋 TABLE: {table_name}")
        print("   Columns:")
        
        # 2. Get structure
        pragma_query = f"PRAGMA table_info({table_name});"
        columns = conn.execute(pragma_query).fetchall()
        
        if not columns:
            print("     (no columns)")
            continue
        
        for col in columns:
            cid, name, col_type, notnull, default, pk = col
            pk_mark = " (PK)" if pk else ""
            null_mark = " NOT NULL" if notnull else ""
            default_str = f" DEFAULT {default}" if default is not None else ""
            print(f"     • {name:<30} | Type: {col_type:<12} {null_mark}{default_str}{pk_mark}")
        
        # 3. Quick row count
        count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
        print(f"     → {count} rows\n")
    
    conn.close()
    print("✅ Inspection complete.")

# Run it
if __name__ == "__main__":
    # Change this path!
    DB_PATH = Path("D:\\Image Data Files SQL\\2024-09-07 Eastern View vs Hugoenot Football.db")  # ← or whatever your file is called
    
    inspect_database(DB_PATH)
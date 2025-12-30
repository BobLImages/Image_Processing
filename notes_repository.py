# notes_repository.py

import sqlite3
import pandas as pd
from pathlib import Path

class NotesRepository:
    def __init__(self, note_db_path: Path):
        print(f'note_db_path = {note_db_path}')
        self.note_db_path = note_db_path
        # self._ensure_file()
        self._ensure_table()

    def _connect(self):
        print(f'Full Note Name:{self.note_db_path}')
        return sqlite3.connect(self.note_db_path)



    def _ensure_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_date TEXT,
                    category TEXT,
                    db_name TEXT,
                    image_name TEXT,
                    note_text TEXT
                )
            """)
            conn.commit()

    def add(self, note_record: dict):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO notes (
                    entry_date,
                    category,
                    db_name,
                    image_name,
                    note_text
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                note_record["entry_date"],
                note_record["category"],
                note_record["db_name"],
                note_record["image_name"],
                note_record["note_text"],
            ))
            conn.commit()

    def load_all(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM notes", conn)

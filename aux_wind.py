# aux_wind.py

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime


class AuxWind:
    """
    A self-contained modal window for adding Notes.
    Accepts a callback for saving the note.
    """
    def __init__(self, parent, save_callback):
        self.parent = parent
        self.save_callback = save_callback

        # --- Create Toplevel window ---
        self.top = tk.Toplevel(parent)
        self.top.title("Notes")
        self.top.transient(parent)   # Keep on top of parent
        self.top.grab_set()          # Modal behavior
        self.top.geometry("1100x300")

        # --- Notes Frame ---
        notes_frame = tk.Frame(self.top, bg="#2d2d2d")
        notes_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Category Dropdown
        tk.Label(notes_frame, text="Category:", fg="#00ff41", bg="#2d2d2d").pack(side="left", padx=(0,5))
        self.note_category_var = tk.StringVar()
        self.note_category_combo = ttk.Combobox(
            notes_frame,
            textvariable=self.note_category_var,
            values=["Image Note", "To-Do", "Gizmo"],
            state="readonly",
            width=15
        )
        self.note_category_combo.pack(side="left", padx=(0,10))
        self.note_category_combo.set("Image Note")  # default

        # Timestamp
        self.note_timestamp_var = tk.StringVar()
        self.note_timestamp_var.set(datetime.now().strftime("%Y-%m-%d %H:%M"))
        tk.Label(notes_frame, textvariable=self.note_timestamp_var,
                 fg="#00ff41", bg="#2d2d2d").pack(side="left", padx=(0,10))

        # Text area
        self.note_text = ScrolledText(notes_frame, width=60, height=4, wrap="word",
                                      bg="#1a1a1a", fg="#ffffff")
        self.note_text.pack(side="left", padx=(0,10))

        # Save Button
        btn_style = {"font": ("Helvetica", 10, "bold"), "bg": "#00ff41", "fg": "black"}
        save_btn = tk.Button(notes_frame, text="Save Note", command=self._save_note, **btn_style)
        save_btn.pack(side="left")

        # Optional: Cancel button
        cancel_btn = tk.Button(notes_frame, text="Cancel", command=self._close, **btn_style)
        cancel_btn.pack(side="left", padx=(5,0))

        # Wait until window is closed (optional if you want blocking modal)
        self.top.wait_window(self.top)

    

    def _save_note(self):
        """Calls the provided callback and closes the window."""
        
        cat_dt_txt_payload = {
            "category": self.note_category_var.get(),
            "entry_date": self.note_timestamp_var.get(),
            'note': self.note_text.get("1.0", tk.END).strip()
            
        }
        # Call controller's save function
        
        self.save_callback(cat_dt_txt_payload)
        self._close()

    def _close(self):
        """Destroy the Instance & Toplevel window."""
        self.notes_win = None
        self.top.destroy()

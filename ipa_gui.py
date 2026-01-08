# ipa_gui.py


import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
from mask_factory import MaskFactory
from mask_defs import MASK_DEFS
from datetime import datetime
import time
# print("MASK_DEFS present:", isinstance(MASK_DEFS, dict), "count:", len(MASK_DEFS))




class ImageAppUI:
    # --- GUI Configuration Constants (Ensure these are present) ---
    CANVAS_WIDTH = 1360
    CANVAS_HEIGHT = 720
    CANVAS_CENTER_X = 700
    CANVAS_CENTER_Y = 450
    # -----------------------------------


    def __init__(self, root, controller):
        self.root = root
        self.ctrl = controller
        self._build()
    

        # THE FOUR HORSEMEN — BINDINGS OF TRUTH
        self.root.bind("<Right>", self.ctrl.go_next_image)
        self.root.bind("<Left>",  self.ctrl.go_prev_image)
        self.root.bind("<Home>",  self.ctrl.go_first_image)
        self.root.bind("<End>",   self.ctrl.go_last_image)

        # Bonus Emperor keys (optional but glorious)
        self.root.bind("n", self.ctrl.go_next_image)
        self.root.bind("p", self.ctrl.go_prev_image)
        self.root.bind("f", self.ctrl.go_first_image)
        self.root.bind("l", self.ctrl.go_last_image)    

    def _build(self):
        self._setup_header()
        self._setup_canvas()
        self._setup_controls()

    def _setup_header(self):
        header = tk.Frame(self.root, bg="#2d2d2d", height=60)
        header.pack(fill="x")
        tk.Label(header, text="IMAGE PROCESSING APP", font=("Helvetica", 16, "bold"),
                 fg="#00ff41", bg="#2d2d2d").pack(pady=15)

    def _setup_canvas(self):
        main = tk.Frame(self.root, bg="#1a1a1a")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(main, bg="#0f0f0f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Create default PIL image ---
        self.current_pil_image = self.create_default_image()
        self.tk_image = ImageTk.PhotoImage(self.current_pil_image)

        # --- Display it on the canvas ---
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
    def create_default_image(self):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), color=(80, 80, 80))  # gray
        draw = ImageDraw.Draw(img)
        draw.text((self.CANVAS_CENTER_X - 50, self.CANVAS_CENTER_Y), "No Image", fill=(150, 150, 150))
        return img

    def _setup_controls(self):
        controls = tk.Frame(self.root, bg="#2d2d2d", height=100)
        controls.pack(fill="x", padx=20, pady=10)
        controls.pack_propagate(False)

        btn_style = {
            "font": ("Helvetica", 10, "bold"), "width": 15, "height": 2,
            "bg": "#00ff41", "fg": "black", "relief": "raised"
        }

        # CHOOSE FOLDER
        self.choose_btn = tk.Button(controls, text="CHOOSE\nFOLDER",
                                    command=self.ctrl.select_directory, **btn_style)
        self.choose_btn.pack(side="left", padx=10)
        
       # === ADD THIS LINE AT THE VERY END OF THIS METHOD ===
        self.choose_btn.focus_set() 

        # ===================================================
        # MASK DROPDOWN
        self.mask_var = tk.StringVar()
        self.mask_combo = ttk.Combobox(controls, textvariable=self.mask_var,
                                       values=[], state="readonly", font=("Helvetica", 12))
        self.mask_combo.pack(side="left", padx=10)
       # ===================================================

       # ===================================================
       # APPLY MASK
        self.apply_btn = tk.Button(controls, text="APPLY MASK",
                                   command=self.ctrl.apply_mask, **btn_style, state="disabled")
        self.apply_btn.pack(side="left", padx=10)

        # SHOW ORIGINAL
        self.original_btn = tk.Button(controls, text="SHOW ORIGINAL",
                                      command=self.ctrl.show_original, **btn_style, state="disabled")
        self.original_btn.pack(side="left", padx=10)



        self.raw_mask_btn = tk.Button(
            controls, text="SHOW RAW MASK",
            command=self.ctrl.show_raw_mask,
            **btn_style, state="disabled"
        )
        self.raw_mask_btn.pack(side="left", padx=10)
        
        self.update_mask_list(list(MASK_DEFS.keys()))


        # Spacer to push right-side buttons
        spacer = tk.Frame(controls)
        spacer.pack(side="left", expand=True, fill='x')

        # NOTES — right side
        self.notes_btn = tk.Button(
            controls,
            text="NOTES",
            command=self.ctrl.open_notes_window,
            **btn_style
        )
        self.notes_btn.pack(side="right", padx=10)

        # DELETE DB — far right (after Notes)
        self.delete_db_btn = tk.Button(
            controls,
            text="DELETE DB",
            command=self.ctrl.delete_current_db,
            bg="#ff4444", fg="white", font=("Helvetica", 10, "bold"),  # red for danger
            relief="raised"
        )
        self.delete_db_btn.pack(side="right", padx=20)









        # # NOTES BUTTON
        # self.notes_btn = tk.Button(
        #     controls,
        #     text="NOTES",
        #     command=self.ctrl.open_notes_window,
        #     **btn_style
        # )
        
        # # --- Push remaining space to separate action buttons from Notes ---
        # spacer = tk.Frame(controls)
        # spacer.pack(side="left", expand=True, fill='x')

        # # --- NOTES BUTTON — now far right ---
        # self.notes_btn = tk.Button(
        #     controls,
        #     text="NOTES",
        #     command=self.ctrl.open_notes_window,
        #     **btn_style
        # )
        # self.notes_btn.pack(side="right", padx=20)        
        # self.notes_btn.pack(side="left", padx=10)        
                

        # STATUS BAR
        self.status = tk.Label(self.root, text="Ready", fg="#00ff41", bg="#1a1a1a",
                               anchor="w", font=("Helvetica", 9))
        self.status.pack(fill="x", padx=20, pady=5)


    def update_mask_list(self, names):
        self.mask_combo['values'] = names
        if names and not self.mask_var.get():
            self.mask_var.set(names[0])

    def enable_mask_buttons(self):
        self.apply_btn.config(state="normal")
        self.original_btn.config(state="normal")
        self.raw_mask_btn.config(state= 'normal')


    def show_image(self, pil_image: Image.Image):
        """
        Displays a PIL image on the canvas and updates the internal reference.
        """
        # --- CRITICAL: Store the original, full-size image object ---
        self.current_pil_image = pil_image.copy() # Store the full size original

        # Resize for canvas display (preserve aspect ratio)
        # print(f' Displaying image')
        canvas_w, canvas_h = self.CANVAS_WIDTH, self.CANVAS_HEIGHT 
        
        # --- FIX IS HERE ---
        # 1. Apply thumbnail directly to the temporary copy
        pil_image.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        
        # 2. Convert the *resized* PIL image to a Tkinter format
        tk_img = ImageTk.PhotoImage(pil_image)
        # --- END FIX ---
        
        # print(f'converting to tkinter')
        
        # Center on canvas
        self.canvas.create_image(
            self.CANVAS_CENTER_X, self.CANVAS_CENTER_Y,
            anchor="center",
            image=tk_img
        )
        # Keep a reference to prevent garbage collection
        self.canvas.image = tk_img
        self.status.config(text="Image displayed on canvas.")

    def set_status(self, text):
        self.status.config(text=text)


    def _setup_notes_panel(self, notes_win):
        """
        Sets up a Notes panel with category, timestamp, and text entry.
        """
        notes_frame = tk.Frame(notes_win, bg="#2d2d2d", height=150)
        notes_frame.pack(fill="x", padx=20, pady=10)
        notes_frame.pack_propagate(False)

        # Category dropdown
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

        # Timestamp label
        self.note_timestamp_var = tk.StringVar()
        self.note_timestamp_var.set(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.note_timestamp_label = tk.Label(notes_frame, textvariable=self.note_timestamp_var,
                                             fg="#00ff41", bg="#2d2d2d")
        self.note_timestamp_label.pack(side="left", padx=(0,10))

        # Text area
        self.note_text = ScrolledText(notes_frame, width=80, height=4, wrap="word", bg="#1a1a1a", fg="#ffffff")
        self.note_text.pack(side="left", padx=(0,10))
        



        # Save button
        btn_style = {"font": ("Helvetica", 10, "bold"), "bg": "#00ff41", "fg": "black"}
        self.note_save_btn = tk.Button(notes_frame, text="Save Note", command=self.ctrl.save_note, **btn_style)
        self.note_save_btn.pack(side="left")

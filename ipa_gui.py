# ipa_ui.py


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from mask_factory import MaskFactory, MASK_DEFS # Ensure MASK_DEFS is imported


from navigation import NAVIGATION_MAP

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
        self.root.bind("<Right>", self.ctrl.next_image)
        self.root.bind("<Left>",  self.ctrl.prev_image)
        self.root.bind("<Home>",  self.ctrl.go_to_first)
        self.root.bind("<End>",   self.ctrl.go_to_last)

        # Bonus Emperor keys (optional but glorious)
        self.root.bind("n", self.ctrl.next_image)
        self.root.bind("p", self.ctrl.prev_image)
        self.root.bind("f", self.ctrl.go_to_first)
        self.root.bind("l", self.ctrl.go_to_last)    

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








        # CHOOSE FOLDER
        self.choose_btn = tk.Button(controls, text="CHOOSE\nFOLDER",
                                    command=self.ctrl.select_directory, **btn_style)
        self.choose_btn.pack(side="left", padx=10)

        # MASK DROPDOWN
        self.mask_var = tk.StringVar()
        self.mask_combo = ttk.Combobox(controls, textvariable=self.mask_var,
                                       values=[], state="readonly", font=("Helvetica", 12))
        self.mask_combo.pack(side="left", padx=10)


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


    def show_image(self, display_image: Image.Image):
# (self, display_image: Image.Image):
        """
        Displays a PIL image on the canvas and updates the internal reference.
        """
        # --- CRITICAL: Store the original, full-size image object ---
        self.current_pil_image = display_image.copy() # Store the full size original

        # Resize for canvas display (preserve aspect ratio)
        print(f' Displaying image')
        canvas_w, canvas_h = self.CANVAS_WIDTH, self.CANVAS_HEIGHT 
        
        # --- FIX IS HERE ---
        # 1. Apply thumbnail directly to the temporary copy
        display_image.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        
        # 2. Convert the *resized* PIL image to a Tkinter format
        tk_img = ImageTk.PhotoImage(display_image)
        # --- END FIX ---
        
        print(f'converting to tkinter')
        
        # Center on canvas
        self.canvas.create_image(
            self.CANVAS_CENTER_X, self.CANVAS_CENTER_Y,
            anchor="center",
            image=tk_img
        )
        # Keep a reference to prevent garbage collection
        self.canvas.image = tk_img
        self.status.config(text="Image displayed on canvas.")










    # def show_image(self, img_pil: Image.Image):
    #     img_pil.thumbnail((1360, 720), Image.Resampling.LANCZOS)
    #     img_tk = ImageTk.PhotoImage(img_pil)
    #     self.canvas.delete("all")
    #     self.canvas.create_image(700, 450, anchor="center", image=img_tk)
    #     self.canvas.image = img_tk  # keep reference

    def set_status(self, text):
        self.status.config(text=text)
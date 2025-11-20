# ipa_controller.py

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw

from ipa_gui import ImageAppUI
from image_catalog import ImageCatalog
from data_handler import DataHandler
from mask_factory import MaskFactory, MASK_DEFS 
from typing import Dict, Union



class ImageAppController:
    def __init__(self, root):
#***************************OBJECT ATTRIBUTES******************************
        self.root = root
        self.root.geometry("1600x1200") # <-- Set the default size here
        self.catalog = None
        self.ui = ImageAppUI(root, self)
        self.ui.set_status("Ready. Choose folder.")
        self.current_index = 0


#***************************METHOD START******************************

    def select_directory(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        handler = DataHandler(folder)
        if not handler.db_ready:
            self.ui.set_status("No DB found.")
            return

        df = handler.load_from_db()
        self.catalog = ImageCatalog(df)
        self.catalog.acquire_image(self.current_index)  # load first image

        # Show first image
        pil = self.catalog.pil_img
        
        self.ui.show_image(pil)

        # Update UI
        self.ui.update_mask_list(list(MASK_DEFS.keys()))
        self.ui.enable_mask_buttons()
        self.ui.set_status(f"Loaded {len(df)} images")




    def apply_mask(self):
        name = self.ui.mask_var.get()
        mask_array = MaskFactory.build(name, self.catalog)
        if mask_array is None:
            self.ui.set_status(f"Failed: {name}")
            return

        pil = self.catalog.generate_blended_mask_for_display(name, mask_array)
        self.ui.show_image(pil)
        self.ui.set_status(f"Mask: {name}")


    def show_original(self):
        pil = self.catalog.get_original_pil_image()
        if pil:
            self.ui.show_image(pil)
            self.ui.set_status("Original image")


    def show_raw_mask(self):
        name = self.ui.mask_var.get()
        
        pil_mask = self.catalog.get_pure_binary_mask_display(name)
        if pil_mask is None:
            self.ui.set_status(f"Raw mask '{name}' not available")
            return

        self.ui.show_image(pil_mask)
        self.ui.set_status(f"Raw mask: {name}  (white = active)")


    def next_image(self, event = None):
        """Move to next image and load it."""

        if self.current_index < len(self.catalog.df) - 1:
            self.current_index += 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_display()   
            print(f"First image loaded")

    
    def prev_image(self, event = None):
        """Move to previous image and load it."""
        if self.current_index > 0:
            self.current_index -= 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_display()   
            print(f"Previous image loaded")


    def go_to_first(self, event = None):
        """Jump to first image."""
        if self.current_index != 0:
            self.current_index = 0
            self.catalog.acquire_image(self.current_index)
            self._refresh_display()   
            print(f"First image loaded")

    def go_to_last(self, event = None):
        """Jump to last image."""
        if self.current_index != len(self.catalog.df) - 1:
            self.current_index = len(self.catalog.df) - 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_display()   
            print(f"Last image loaded")


    def _refresh_display(self):
        print("INSIDE REFRESH — THE JUICE IS HERE")
        try:
            pil = self.catalog.get_original_pil_image()
            if not pil:
                print("  → NO PIL — NO JUICE")
                return
            print(f"  → GOT PIL: {pil.size}")
            self.ui.show_image(pil)
            print("  → IMAGE SENT TO CANVAS — SLICE SERVED")
        except Exception as e:
            print(f"  → REFRESH CRASHED: {e}")







#***************************KEYHANDLING METHOD PERHAPS REUSE OR SALVAGE 11/21/2025******************************

    # def _handle_navigation(self, event):
    #     action = NAVIGATION_MAP.get(event.keysym) or NAVIGATION_MAP.get(event.char)
    #     if not action:
    #         print(f"NO ACTION: key='{event.keysym}' char='{event.char}' — not in NAVIGATION_MAP")
    #         return
    #     if action == "first":
    #         self.current_index = 0
    #     elif action == "last":
    #         self.current_index = len(self.catalog.df) - 1
    #     elif isinstance(action, int):
    #         print(f'Inside Instance')
    #         self.current_index = max(0, min(
    #             self.current_index + action,
    #             len(self.catalog.df) - 1
    #         ))
    #     else:
    #         return
    #     # ← THE FINAL BATTLE CRY
    #     print(f"JUICE INCOMING: index={self.current_index}, about to refresh")

    #     self.catalog.acquire_image(self.current_index)
    #     print("  → Image loaded, now REFRESHING...")
    #     self._refresh_display()
    #     print("  → REFRESH CALLED — GIVE ME JUICE!")

    
    #     # Force Tkinter to wake up
    #     self.root.update_idletasks()
    #     print("  → FORCED TKINTER UPDATE — NO MORE SILENCE!")



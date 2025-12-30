# ipa_controller.py
# 1. Standard library
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from datetime import datetime
from pathlib import Path

# 2. Third-party
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageDraw

# 3. Local app imports
from ipa_gui import ImageAppUI
from image_catalog import ImageCatalog, ImageRef
from data_handler import DataHandler
from harvester import Harvester, build_db_2_df
from mask_factory import MaskFactory
from mask_defs import MASK_DEFS
from file_record import EventPathGroup, FrameworkPathGroup
from aux_wind import AuxWind
from notes_repository import NotesRepository
from rpt_dbg_tst import RTD
DB_ROOT = Path(r"D:\Image Data Files sql")  # use raw string or escape backslashes

class ImageAppController:
    def __init__(self, root):
#***************************OBJECT ATTRIBUTES******************************
        self.root = root
        self.root.geometry("1600x1200") # <-- Set the default size here
        self.catalog = None
        self.ui = ImageAppUI(root, self)
        self.ui.set_status("Ready. Choose folder.")
        self.current_index = -1
        self.paths: EventPaths | None = None
        self.framework_paths = FrameworkPathGroup(DB_ROOT)
        '''
        This function returns a framework object that has following properties:
        1. db files location                framework_paths.db_root
        2. notes_file location              framework_paths.notes_root
        and an attribute that seems alot
        like a property or vice-versa:
        3. note db file "proposed or actual" framework_paths.notes_db
        '''
#***************************METHODS START******************************

    def _initialize_catalog_from_df(self, df):
        self.catalog = ImageCatalog(df)
        # Reset self.current_index for a df being loaded
        self.current_index = 0
        
        self.catalog.acquire_image(0)
        self.ui.show_image(self.catalog.pil_img)
        self._complete_ui_setup()

    
    def _get_note_df_ready(self,cat_dt_txt_payload):
        is_image_note = True if cat_dt_txt_payload.get('category') == 'Image Note' else False

        df_ready_dict = {'category': cat_dt_txt_payload.get('category'),
        'entry_date': cat_dt_txt_payload.get('entry_date'),
        'db_name': self.catalog.image_ref.db_name if is_image_note else 'N/A',
        'image_name': self.catalog.image_ref.image_name if is_image_note else 'N/A',
        'note_text': cat_dt_txt_payload.get('note'),
        }
        return df_ready_dict

    
    def _complete_ui_setup(self):
        # 1. Load mask names
        mask_names = list(MASK_DEFS.keys())
        self.ui.update_mask_list(mask_names)

        # 2. Enable all the mask-related buttons
        self.ui.apply_btn.config(state="normal")
        self.ui.original_btn.config(state="normal")
        self.ui.raw_mask_btn.config(state="normal")

        # 3. Update status text
        self.ui.set_status("Masks loaded. Ready.")

        # Optional but likely:
        if mask_names:
            self.ui.mask_var.set(mask_names[0])


    #***************************Call_back METHODS ******************************

    def save_note_from_aux(self, cat_dt_txt_payload):
        #
        print(f'Accepts a note dictionary from AuxWind and saves it to the notes_df.')
        if  not hasattr(self, 'notes_repository'):
            self.notes_repository = NotesRepository(self.framework_paths.notes_db)


        note_record = self._get_note_df_ready(cat_dt_txt_payload)
        self.notes_repository.add(note_record)
        self.ui.set_status("Note saved")


    #***************************Button_press/Mouse_click and Navigation METHODS ******************************
    
    def select_directory(self): # 35 lines
        folder = filedialog.askdirectory()
        if not folder:
            return

        event_path = Path(folder)

        # 1. Confirm a valid folder was chosen, if valid == True create event_pths
        
        if not Harvester.is_valid_event_dir(event_path):
            messagebox.showwarning("No Bueno", "Not enough images")
            return
        else:
            valid_paths = Harvester.find_candidate_images(event_path)
            if not valid_paths:
                print(f'No valid paths were found')
                return
            else:
                self.event_paths = EventPathGroup(
                    event_path=event_path,
                    db_root=self.framework_paths.db_root
                )
 

        # 2. If DB exists → create df else process, create db, create df
        if self.event_paths.catalog_db.exists():
            df = build_db_2_df(self.event_paths.catalog_db)
        else:
            # 4. No DB → ask for harvest
            if not messagebox.askyesno("JUICE TIME?", "Create database and harvest?"):
                return
            # 5. Harvest → creates DB + fills it
            df = Harvester(self.event_paths).run()
        
        self._initialize_catalog_from_df(df)
    

    def delete_current_db(self):
        """Delete the current catalog DB file for a fresh start."""
        if not hasattr(self, 'event_paths') or not self.event_paths.catalog_db.exists():
            self.ui.set_status("No DB to delete")
            return
        
        if messagebox.askyesno("Delete DB", f"Delete {self.event_paths.catalog_db.name}?\nNext run will re-harvest."):
            try:
                self.event_paths.catalog_db.unlink()
                self.ui.set_status("DB deleted — next load will be fresh")
                print(f"Deleted: {self.event_paths.catalog_db}")
            except Exception as e:
                self.ui.set_status(f"Delete failed: {e}")



    def open_notes_window(self):
        if not hasattr(self, 'notes_win') or self.notes_win is None:
            notes_win = AuxWind(self.root, self.save_note_from_aux)
        else:
            self.notes_win.top.lift()  # Bring existing window to front


    def apply_mask(self):
        name = self.ui.mask_var.get()
        four_amigos = {"original":self.catalog.img_org_cv,
                        "sharpened":self.catalog.img_sharp_cv,
                        "hsv":self.catalog.img_hsv,
                        "greyscale":self.catalog.img_gs,
        }
        mask_array = MaskFactory.build(name, four_amigos, self.catalog.repository )
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


    def go_first_image(self, event = None):
        if self.current_index != 0:
            self.current_index = 0
            self.catalog.acquire_image(self.current_index)
            self._refresh_image()   

    
    def go_last_image(self, event = None):
        if self.current_index != len(self.catalog.df) - 1:
            self.current_index = len(self.catalog.df) - 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_image()   



    def go_next_image(self, event = None):
        if self.current_index < len(self.catalog.df) - 1:
            self.current_index += 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_image()   

    
    def go_prev_image(self, event = None):
        if self.current_index > 0:
            self.current_index -= 1
            self.catalog.acquire_image(self.current_index)
            self._refresh_image()   


    def _refresh_image(self):

        try:
            pil = self.catalog.get_original_pil_image()
            if not pil:
                return
            print(f"  → GOT PIL: {pil.size}")
            self.ui.show_image(pil)
        except Exception as e:
            print(f"  → ISSUE REFRESHING IMAGE: {e}")

'''
    ***************************Extraneous material for Notes ******************************

'''

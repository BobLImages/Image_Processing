# harvester.py

import sqlite3
import cv2
import numpy as np
from pathlib import Path

import pandas as pd
from file_record import EventPathGroup, get_disk_files
from rpt_dbg_tst import RTD
from builder import Builder
from color_image import ColorImage



class Harvester:
    def __init__(self,paths: EventPathGroup):
        self.paths = paths

#process_valid_images
    def process_event_images(self):
        color_images = process_image_objs(self.paths)


# @staticmethod
# def _create_db_if_needed(db_file_path):
#     if db_file_path.exists():
#         return

#     conn = sqlite3.connect(db_file_path)
#     cursor = conn.cursor()
#     table = db_file_path.name

#     cursor.execute(f"""
#     CREATE TABLE IF NOT EXISTS [{table}] (
#         image_id          INTEGER PRIMARY KEY,
#         file_path         TEXT NOT NULL,
#         original_width    INTEGER,
#         original_height   INTEGER,
#         ro_width          INTEGER,
#         ro_height         INTEGER,
#         scale_factor      REAL,
#         orientation       TEXT,
#         brightness FLOAT,                         
#         contrast   FLOAT,   
#         laplacian FLOAT,
#         harris_corners FLOAT,  
#         haze_factor FLOAT,  
#         variance FLOAT,  
#         shv FLOAT,  
        


#         classification    TEXT DEFAULT 'U',
#         ro_image          BLOB,
#         denoised_image    BLOB,
#         sharpened_image   BLOB,
#         gray_image        BLOB,
#         notes             TEXT
#     )""")

#     cursor.execute(f"""
#     CREATE TABLE IF NOT EXISTS [{table}--classifications] (
#         image_id       INTEGER,
#         model_name     TEXT NOT NULL,
#         classification TEXT NOT NULL,
#         confidence     REAL,
#         run_date       TEXT DEFAULT (datetime('now')),
#         notes          TEXT,
#         PRIMARY KEY (image_id, model_name),
#         FOREIGN KEY (image_id) REFERENCES [{table}](image_id) ON DELETE CASCADE
#     )""")

#     conn.commit()
#     conn.close()





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


@staticmethod
def convert_image(img_or_blob, direction: str = 'encode'):
    """
    General image converter — encode to BLOB or decode from BLOB.
    
    direction = 'encode' : numpy image → bytes (for DB save)
    direction = 'decode' : bytes → numpy image (for load)
    """
    if direction == 'encode':
        if img_or_blob is None:
            return None
        success, buffer = cv2.imencode('.png', img_or_blob.astype(np.uint8))
        return buffer.tobytes() if success else None
    
    elif direction == 'decode':
        if img_or_blob is None:
            return None
        buf = np.frombuffer(img_or_blob, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print("Warning: Failed to decode BLOB")
        return img
    
    else:
        raise ValueError("direction must be 'encode' or 'decode'")



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


class Harvester:
    def __init__(self,paths: EventPathGroup):
        self.paths = paths

#process_valid_images
    def process_event_images(self):
        color_images = process_image_objs(self.paths)



















		# _run_stats()
		# _create_db_if_needed(self.db_path)
		# _save_colorimages_to_db()       
		




		# return df


		

	# def _run_stat():

	# 	for image in color_images:
	# 		image.laplacian = get_laplacian(image.image_gs)
	# 		image.brightness = get_brightness(image.image_sharp)
	# 		image.contrast = get_contrast(image.image_gs)
	# 		image.harris = get_harris(image.image_gs)
	# 		image.haze_factor = get_haze_factor(image.image_sharp)
	# 		image.variance = get_variance(image.image_gs)
	# 		image.hsv = get_hsv(image.image_gs)
	# 		image.snr = get_snr()



	# def _save_colorimages_to_db(self):
	# 	conn = sqlite3.connect(self.db_file_path)
	# 	cursor = conn.cursor()
	# 	table = db_file_path.name


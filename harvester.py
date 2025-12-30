# harvester.py

import sqlite3
import cv2
import numpy as np
from pathlib import Path
from color_image import ColorImage
import pandas as pd
from file_record import EventPathGroup
from rpt_dbg_tst import RTD





@staticmethod
def is_valid_event_dir(event_dir: Path) -> bool:
    valid = Harvester.find_candidate_images(event_dir)
    return len(valid) >= 5


@staticmethod
def find_candidate_images(event_path: Path) -> list[Path]:
    	# Sacred filter — only originals
     return [
        p for p in event_path.glob("*.[jJ][pP][gG]")
        if p.is_file()
           and not p.name.lower().startswith('r_')
           and '$' not in p.name and "exposure" not in p.name
    ]

@staticmethod
def populate_color_images(image_paths):
	color_images = []
	for idx, path in enumerate(image_paths, 1):
		ro_image,geometry_dict = geometry(path)

		image = ColorImage(
		ro_image,
		idx,
		path,
		geometry_dict
		)
		color_images.append(image)
	return color_images


@staticmethod
def build_db_2_df(catalog_db: Path) -> pd.DataFrame | None:
    if not catalog_db.exists():
        print(f"DB not found: {catalog_db}")
        return None

    try:
        with sqlite3.connect(catalog_db) as conn:
            df = pd.read_sql("SELECT * FROM images", conn)

            # Decode BLOBs
            df['Original_Image'] = df['Original_Image'].apply(_decode_image)
            df['Sharpened_Image'] = df['Sharpened_Image'].apply(_decode_image)
            df['Grayscale_Image'] = df['Grayscale_Image'].apply(_decode_image)

            # Convert File_Name back to Path
            df['File_Name'] = df['File_Name'].apply(lambda s: Path(s) if s else None)

        print("DF successfully loaded from DB")
        print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"Failed to load DB: {e}")
        return None    


@staticmethod
def _decode_image(blob):
    if blob is None:
        return None
    buf = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        print("Warning: Failed to decode image from BLOB")
    return img



@staticmethod
def geometry(path: Path):
	img = cv2.imread(str(path))
	if img is None:
	    print(f"Failed to load: {path}")
	    return None, None

	h, w = img.shape[:2]
	scale = 896 / w if w < h else 2016 / w if w > h else 1.0
	ro_w = int(w * scale)
	ro_h = int(h * scale)
	ro_image = cv2.resize(img, (ro_w, ro_h), interpolation=cv2.INTER_AREA)

	geometry_dict = {
		"original_width": w,
		"original_height": h,
		"ro_width": ro_w,
		"ro_height": ro_h,
		"scale_factor": scale,
		"orientation": "Portrait" if h > w else "Landscape",
		}
	return ro_image, geometry_dict

@staticmethod
def build_df_2_db(catalog_db:Path, df:pd.DataFrame):
		 # === SAVE TO DB — FULL AND FINAL ===

		conn = sqlite3.connect(catalog_db)

		# Conversion - images to BLOBs,paths to strings
		save_df = df.copy()
		save_df['Original_Image'] = save_df['Original_Image'].apply(_encode_image)
		save_df['Sharpened_Image'] = save_df['Sharpened_Image'].apply(_encode_image)
		save_df['Grayscale_Image'] = save_df['Grayscale_Image'].apply(_encode_image)
		save_df['File_Name'] = save_df['File_Name'].apply(lambda p: str(p) if p else None)

		# Save — this creates the table if missing
		save_df.to_sql('images', conn, if_exists='replace', index=False)
		conn.commit()
		conn.close()

		print(f"SAVED {len(df)} images to {catalog_db}")


@staticmethod
def _encode_image(img):
    if img is None:
        return None
    success, buffer = cv2.imencode('.png', img.astype(np.uint8))
    return buffer.tobytes() if success else None



@staticmethod
def build_objs_2_df(color_images):
    # Define all columns in your final DataFrame
    columns = [
        'Image_ID', 'File_Name', 'Brightness', 'Contrast',
        'Laplacian','Original_Image', 'Grayscale_Image', 'Sharpened_Image'
    ]

    data_list = []
    for img in color_images:
        data_list.append({
            'Image_ID': img.image_id,
            'File_Name': img.image_path,
			'Brightness': img.brightness,
			'Contrast': img.contrast,            
			'Laplacian': img.laplacian,           
 			'Sharpened_Image': img.RODS_image,
			'Original_Image': img.ro_image,
			'Grayscale_Image': img.image_gs,
            # Everything else is None for now
            # **{col: None for col in columns if col not in ['Image_ID', 'File_Name', 'Geometry', 'Sharpened_Image','Grayscale_Image','Original_Image', 'Contrast', 'Brightness','Laplacian']}
        })

    images_pd = pd.DataFrame(data_list, columns=columns)
    RTD.report(images_pd,10)
    return images_pd


@staticmethod
def _create_db_if_needed(db_file_path):
    if db_file_path.exists():
        return

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    table = db_file_path.name

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS [{table}] (
        image_id          INTEGER PRIMARY KEY,
        file_path         TEXT NOT NULL,
        file_name         TEXT NOT NULL,
        ro_image          BLOB,
        gray_image        BLOB,
        denoised_image    BLOB,
        sharpened_image   BLOB,
        original_width    INTEGER,
        original_height   INTEGER,
        ro_width          INTEGER,
        ro_height         INTEGER,
        scale_factor      REAL,
        orientation       TEXT,
        exif              TEXT,
        iptc              TEXT,
        stats             TEXT,
        classification    TEXT DEFAULT 'U',
        review_flag       INTEGER DEFAULT 0,
        notes             TEXT
    )""")

    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS [{table}--classifications] (
        image_id       INTEGER,
        model_name     TEXT NOT NULL,
        classification TEXT NOT NULL,
        confidence     REAL,
        run_date       TEXT DEFAULT (datetime('now')),
        notes          TEXT,
        PRIMARY KEY (image_id, model_name),
        FOREIGN KEY (image_id) REFERENCES [{table}](image_id) ON DELETE CASCADE
    )""")

    conn.commit()
    conn.close()


class Harvester:
	def __init__(self,paths: EventPathGroup):
		self.paths = paths
	

	def run(self):
		image_paths = find_candidate_images(self.paths.event_path)
		color_images = populate_color_images(image_paths)
		# test_slideshow(self.color_images)
		df = build_objs_2_df(color_images)
		build_df_2_db(self.paths.catalog_db,df)
		df = build_db_2_df(self.paths.catalog_db)
		



		# === DEBUG: Print what we just saved ===
		# print("\n=== DB CONTENTS AFTER SAVE ===")
		# debug_df = pd.read_sql('SELECT Image_ID, File_Name, Brightness, Contrast, Laplacian FROM images LIMIT 10', sqlite3.connect(db_path))
		# print(debug_df.to_string(index=False))
        
		# # Optional: quick stats
		# print("\nNumeric summary:")
		# print(debug_df[['Brightness', 'Contrast', 'Laplacian']].describe().round(2))
    




		# print(debug_df.to_string(index=False))

		# # Optional: show column types
		# print("\nColumn dtypes:")
		# print(debug_df.dtypes)

		# # Optional: show numeric summary
		# numeric_cols = debug_df.select_dtypes(include=['float64', 'int64']).columns
		# if len(numeric_cols) > 0:
		#     print("\nNumeric stats summary:")
		#     print(debug_df[numeric_cols].describe())

		# print("=== END DEBUG ===\n")




		return df

	

	@staticmethod
	def _encode_image(img):
	    if img is None:
	        return None
	    success, buffer = cv2.imencode('.png', img.astype(np.uint8))
	    return buffer.tobytes() if success else None






	@staticmethod
	def is_valid_event_dir(event_dir: Path) -> bool:
	    valid = Harvester.find_candidate_images(event_dir)
	    return len(valid) >= 5


	@staticmethod
	def find_candidate_images(event_path: Path) -> list[Path]:
	    	# Sacred filter — only originals
	     return [
	        p for p in event_path.glob("*.[jJ][pP][gG]")
	        if p.is_file()
	           and not p.name.lower().startswith('r_')
	           and '$' not in p.name and "exposure" not in p.name
	    ]






















		# _run_stats()
		# _create_db_if_needed(self.db_path)
		# _save_colorimages_to_db()       
		




		# return df


		

	def _run_stat():

		for image in color_images:
			image.laplacian = get_laplacian(image.image_gs)
			image.brightness = get_brightness(image.image_sharp)
			image.contrast = get_contrast(image.image_gs)
			image.harris = get_harris(image.image_gs)
			image.haze_factor = get_haze_factor(image.image_sharp)
			image.variance = get_variance(image.image_gs)
			image.hsv = get_hsv(image.image_gs)
			image.snr = get_snr()



	def _save_colorimages_to_db(self):
		conn = sqlite3.connect(self.db_file_path)
		cursor = conn.cursor()
		table = db_file_path.name


# builder.py

from pathlib import Path
import pandas as pd
from file_record import get_disk_files
from color_image import ColorImage
from rpt_dbg_tst import RTD
import sqlite3
import cv2
import numpy as np
from image_statistics import ImageStatistics



class Builder:

	@staticmethod
	def _decode_image(blob):
		if blob is None:
			return None
		buf = np.frombuffer(blob, np.uint8)
		img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
		if img is None:
			print("Warning: Failed to decode image from BLOB")
		return img


	@staticmethod
	def _encode_image(img):
		if img is None:
			return None
		success, buffer = cv2.imencode('.png', img.astype(np.uint8))
		return buffer.tobytes() if success else None



	@staticmethod
	def build_raw_2_obj(paths):
		image_paths = is_valid_event_dir(paths.event_path)
		if not image_paths:
			return None
		else:
			objs = populate_color_images(image_paths) 
		if not objs:
			return None
		else:
			return objs



	@staticmethod
	def build_objs_2_df(color_images):

		# Define all columns in your final DataFrame
		columns = [
			'Image_ID', 'File_Name',
			'Geometry', 'Image_Stats', 'EXIF_Stats',
			'Original_Image','Denoised_Image', 'Sharpened_Image', 'Grayscale_Image',
			'classification', 'review_flag', 'notes',
		]

		data_list = []
		for img in color_images:
				data_list.append({
				'Image_ID': img.image_id,
				'File_Name': img.image_path,
				'Geometry': img.geometry,
	            'Image_Stats': img.image_stats,
	 			'EXIF_Stats': img.exif_stats,
				'Original_Image': img.ro_image,
				'Denoised_Image': img.rod_image,
				'Sharpened_Image': img.RODS_image,
				'Grayscale_Image': img.image_gs,
				'classification': 'U',
				'review_flag':0,
				'notes': None,
				})

		images_pd = pd.DataFrame(data_list, columns=columns)
		return images_pd


	@staticmethod
	def build_df_2_db(catalog_db: Path, df: pd.DataFrame) -> bool:
		if df.empty:
			print("DF empty — nothing to save")
			return False
		temp_df = df.copy()
		converted_df = encode_df(temp_df)
		del temp_df
		catalog_db.parent.mkdir(parents=True, exist_ok=True)
		
		conn = sqlite3.connect(catalog_db)
		converted_df.to_sql('images', conn, if_exists='replace', index=False)
				            
		# Optional: verify row count
		cursor = conn.cursor()
		cursor.execute("SELECT COUNT(*) FROM images")
		row_count = cursor.fetchone()[0]
		print(f"SAVED {len(converted_df)} rows → verified {row_count} in DB")
		return True
		if conn:
			conn.commit()
			conn.close()



	@staticmethod
	def build_db_2_df(catalog_db: Path) -> pd.DataFrame | None:
		# Ensure db and get all tables

		if not catalog_db.exists():
			print(f"DB not found: {catalog_db}")
			return None
		conn = sqlite3.connect(catalog_db)
		tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
    
	    # Step 1: Prefer ver2 "images" table
		if "images" in tables:
			print("Found ver2 'images' table → loading")
    
	    # Step 2: Look for legacy signature in ANY table
		else:
			legacy_table = None
			for table in tables:
				cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
				if "Black_Pixels" in cols:  # ← our baby, if it exists
					legacy_table = table
					break
	            # Optional: add more checks, e.g., if "SomeOtherOldColumn" in cols
	        
			if legacy_table:
				print(f"Found legacy data in table '{legacy_table}' → creating 'images' table")
				create_images_tbl_ver2(conn)
				convert_db_ver2(conn, legacy_table)  # pass table name
			else:
				print("No legacy or ver2 data found → starting fresh")
	            # df = pd.DataFrame(columns=MODERN_COLUMNS)
		try:
			conn = sqlite3.connect(catalog_db)
			df = pd.read_sql("SELECT * FROM images", conn)
		except Exception as e:
			print(f"Failed to load DataFrame: {e}")
			return None    
		finally:
			if conn:
				conn.close
			print("Raw df successfully loaded from DB")
			return decode_db(df)	    		



@staticmethod
def decode_db(df:pd.DataFrame):
	df_out = df.copy()	    

	# 1. Convert File_Name back to Path
	df_out['File_Name'] = df_out['File_Name'].apply(lambda s: Path(s) if s else None)

	# 2. Decode BLOBs back to numpy arrays
	image_cols = ['Original_Image', 'Denoised_Image', 'Sharpened_Image', 'Grayscale_Image']
	for col in image_cols:
		if col in df_out.columns:
			df_out[col] = df_out[col].apply(Builder._decode_image)

	# 3. Rebuild geometry dict from flattened columns
	geometry_cols = ['original_width', 'original_height', 'ro_width', 'ro_height',
						'scale_factor', 'orientation']
	if all(col in df_out.columns for col in geometry_cols):
		def rebuild_geometry(row):
			return {
				'original_width': row['original_width'],
				'original_height': row['original_height'],
				'ro_width': row['ro_width'],
				'ro_height': row['ro_height'],
				'scale_factor': row['scale_factor'],
				'orientation': row['orientation'],
            }
			df_out['Geometry'] = df_out.apply(rebuild_geometry, axis=1)

	# 4. Rebuild Image_Stats dataclass from flattened stats
		stats_cols = ['brightness', 'contrast', 'laplacian', 'harris_corners',
					'haze_factor', 'variance', 'shv']
		if all(col in df_out.columns for col in stats_cols):
			def rebuild_stats(row):
				return ImageStatistics(
				brightness=row['brightness'],
				contrast=row['contrast'],
				laplacian=row['laplacian'],
				harris_corners=row['harris_corners'],
				haze_factor=row['haze_factor'],
				variance=row['variance'],
				shv=row['shv'],
		)
	df_out['Image_Stats'] = df_out.apply(rebuild_stats, axis=1)
	# Optional: drop the flattened columns if you want a clean "rich" view
	df_out = df_out.drop(columns=geometry_cols + stats_cols)
	return df_out

@staticmethod
def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()

    # 1. file_path: Path → str
    if 'File_Name' in df_out.columns:
        df_out['File_Name'] = df_out['File_Name'].apply(
            lambda x: str(x) if isinstance(x, Path) else x
        )

    # 2. Images → BLOB (using sqlite3.Binary for safety)
    image_cols = ['Original_Image', 'Denoised_Image', 'Sharpened_Image', 'Grayscale_Image']
    for col in image_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(Builder._encode_image)

    # 3. geometry dict → explicit flattened columns
    if 'Geometry' in df_out.columns:
        g = df_out['Geometry']
        df_out['original_width']   = g.apply(lambda d: d.get('original_width')   if d else None)
        df_out['original_height']  = g.apply(lambda d: d.get('original_height')  if d else None)
        df_out['ro_width']         = g.apply(lambda d: d.get('ro_width')         if d else None)
        df_out['ro_height']        = g.apply(lambda d: d.get('ro_height')        if d else None)
        df_out['scale_factor']     = g.apply(lambda d: d.get('scale_factor')     if d else None)
        df_out['orientation']      = g.apply(lambda d: d.get('orientation')      if d else None)
        df_out = df_out.drop(columns=['Geometry'])

    # 4. image_stats dataclass → explicit flattened columns
    if 'Image_Stats' in df_out.columns:
        s = df_out['Image_Stats']
        df_out['brightness']       = s.apply(lambda obj: obj.brightness       if obj else None)
        df_out['contrast']         = s.apply(lambda obj: obj.contrast         if obj else None)
        df_out['laplacian']        = s.apply(lambda obj: obj.laplacian        if obj else None)
        df_out['harris_corners']   = s.apply(lambda obj: obj.harris_corners   if obj else None)
        df_out['haze_factor']      = s.apply(lambda obj: obj.haze_factor      if obj else None)
        df_out['variance']         = s.apply(lambda obj: obj.variance         if obj else None)
        df_out['shv']              = s.apply(lambda obj: obj.shv              if obj else None)
        df_out = df_out.drop(columns=['Image_Stats'])


    # 5. EXIF_stats dataclass → explicit flattened columns
    if 'EXIF_Stats' in df_out.columns:
        t = df_out['EXIF_Stats']
        df_out['exposure']       = t.apply(lambda obj: obj.exposure      if obj else None)
        df_out['f_stop']         = t.apply(lambda obj: obj.f_stop         if obj else None)
        df_out['iso']        = t.apply(lambda obj: obj.iso        if obj else None)
        df_out['focal_length']   = t.apply(lambda obj: obj.focal_length   if obj else None)
        df_out = df_out.drop(columns=['EXIF_Stats'])

        # === FUTURE STATS GO HERE ===


    # Ensure app columns exist and force final column order
    final_order = [
        'Image_ID',
        'File_Name',
        'original_width', 'original_height', 'ro_width', 'ro_height',
        'scale_factor', 'orientation',
        # Image_Stats
        'brightness', 'contrast', 'laplacian', 'harris_corners',
        'haze_factor', 'variance', 'shv',
        #EXIF_Stats
        'exposure','f_stop','iso','focal_length',
        # Images (big BLOBs last)
		'Original_Image', 'Denoised_Image', 'Sharpened_Image', 'Grayscale_Image',
		# App state
        'classification', 'review_flag', 'notes',
    ]

    # Add defaults for app fields if missing
    if 'classification' not in df_out.columns:
        df_out['classification'] = 'U'
    if 'review_flag' not in df_out.columns:
        df_out['review_flag'] = 0
    if 'notes' not in df_out.columns:
        df_out['notes'] = None

    df_out = df_out.reindex(columns=final_order)
    return df_out


@staticmethod
def convert_db_ver2(conn, legacy_table="Sheet_1"):
    """
    Migrates legacy data (e.g., Sheet_1) → modern 'images' table (v2 schema).
    Handles case differences explicitly, adds new fields, drops obsolete junk.
    """
    print(f"\n✨ Starting v2 migration from legacy table '{legacy_table}'...")

    # Count rows first for confirmation
    row_count = conn.execute(f"SELECT COUNT(*) FROM [{legacy_table}]").fetchone()[0]
    print(f"   Found {row_count} rows in '{legacy_table}'")

    # Ensure the modern 'images' table exists (you should have this defined elsewhere)
    # create_images_table_v2(conn)   # ← uncomment if needed

    # The magic SQL — your down-and-dirty masterpiece
    migrate_sql = f"""
    INSERT INTO images (
        Image_ID, File_Name,
        original_width, original_height, ro_width, ro_height,
        scale_factor, orientation,
        brightness, contrast, laplacian, harris_corners,
        haze_factor, variance, shv,
        exposure, f_stop, iso, focal_length,
        Original_Image, Denoised_Image, Sharpened_Image, Grayscale_Image,
        classification, review_flag, notes
    )
    SELECT 
        Image_ID, File_Name,
        Original_Width  AS original_width,
        Original_Height AS original_height,
        NULL            AS ro_width,
        NULL            AS ro_height,
        NULL            AS scale_factor,
        Orientation     AS orientation,
        Brightness      AS brightness,
        Contrast        AS contrast,
        Laplacian       AS laplacian,
        Harris_Corners  AS harris_corners,
        Haze_Factor     AS haze_factor,
        Variance        AS variance,
        SHV             AS shv,
        Exposure        AS exposure,
        F_Stop          AS f_stop,
        ISO             AS iso,
        Focal_Length    AS focal_length,
        Original_Image,
        Denoised_Image,
        Sharpened_Image,
        Grayscale_Image,
        Classification  AS classification,
        0               AS review_flag,
        ''              AS notes
    FROM [{legacy_table}];
    """

    try:
        conn.execute(migrate_sql)
        conn.commit()
        print(f"✨ Magic complete! {row_count} rows successfully migrated to 'images' table.")
        print(f"   Legacy table '{legacy_table}' is still there — safe and sound.")
        print(f"   When you're 100% happy, run: DROP TABLE [{legacy_table}];")
    except Exception as e:
        conn.rollback()
        print(f"⚠️  Something went wrong: {e}")
        print("   No changes were made (rollback complete).")

    print("\nDone. Go check your 'images' table — it should be looking sharp now.\n")
    return row_count




@staticmethod
def create_images_tbl_ver2(conn):
    print("Creating ver2 schema  'images' table...")
    
    create_sql = """
    CREATE TABLE IF NOT EXISTS images (
        Image_ID INTEGER PRIMARY KEY,
        File_Name TEXT,
        original_width INTEGER,
        original_height INTEGER,
        ro_width INTEGER,
        ro_height INTEGER,
        scale_factor REAL,
        orientation TEXT,
        brightness REAL,
        contrast REAL,
        laplacian REAL,
        harris_corners INTEGER,
        haze_factor REAL,
        variance REAL,
        shv REAL,
        exposure REAL,
        f_stop REAL,
        iso INTEGER,
        focal_length REAL,          -- changed to REAL in case it's fractional
        Original_Image BLOB,
        Denoised_Image BLOB,
        Sharpened_Image BLOB,
        Grayscale_Image BLOB,
        classification TEXT DEFAULT '',
        review_flag INTEGER DEFAULT 0,
        notes TEXT DEFAULT ''
    );
    """
    conn.execute(create_sql)
    conn.commit()
    print("   'images' table ready for v2 data!")
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
			'Image_ID', 'File_Name', 'Geometry', 'Image_Stats', 'Original_Image','Denoised_Image', 'Sharpened_Image', 'Grayscale_Image'
		]

		data_list = []
		for img in color_images:
				data_list.append({
				'Image_ID': img.image_id,
				'File_Name': img.image_path,
				'Geometry': img.geometry,
	            'Image_Stats': img.image_stats,
	 			# 'EXIF_Stats': image.exif_stats
				'Original_Image': img.ro_image,
				'Denoised_Image': img.rod_image,
				'Sharpened_Image': img.RODS_image,
				'Grayscale_Image': img.image_gs,
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
		if not catalog_db.exists():
			print(f"DB not found: {catalog_db}")
			return None
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

        # === FUTURE STATS GO HERE ===

        df_out = df_out.drop(columns=['Image_Stats'])

    # Ensure app columns exist and force final column order
    final_order = [
        'Image_ID',
        'File_Name',
        'original_width', 'original_height', 'ro_width', 'ro_height',
        'scale_factor', 'orientation',
        # Stats
        'brightness', 'contrast', 'laplacian', 'harris_corners',
        'haze_factor', 'variance', 'shv',
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


# color_image.py

import cv2
import numpy as np
from image_functions import ImageFunctions as IF
from image_statistics import ImageStatistics
from pathlib import Path
from rpt_dbg_tst import RTD



class ColorImage:
	def __init__(self, r_src, image_id=None, fname=None, geometry=None):
		self.image_id = image_id
		self.image_path = fname

		# Handle geometry cleanly
		self.geometry = geometry 
		self.classification = "U"
		
		# Preprocessing
		self.ro_image = r_src
		self.rod_image = cv2.fastNlMeansDenoisingColored(self.ro_image.astype(np.uint8), None, 2, 2, 7, 21)
		self.RODS_image = IF.sharpen(self.rod_image)
		self.image_gs = cv2.cvtColor(self.RODS_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		self.image_stats = ImageStatistics(
		    brightness = IF.get_brightness(self.RODS_image),
		    contrast = IF.get_contrast(self.image_gs),
		    laplacian = IF.get_laplacian(self.image_gs),
		    harris_corners = IF.get_harris(self.image_gs),
		    haze_factor = IF.get_haze_factor(self.RODS_image),
		    variance = IF.get_variance(self.image_gs),
		    shv = IF.get_shv(self.image_gs)
			)


	@staticmethod
	def process_image_objs(paths:list[Path]):
		color_images = []
		for idx, path in enumerate(paths, 1):
			ro_image,geometry_dict = IF.geometry(path)
	        
			image = ColorImage(
			ro_image,
			idx,
			path,
			geometry_dict
			)

			color_images.append(image)
			print(f"Processing Index: {image.image_id}  File: {image.image_path.name}")

		title = f'Inside process color_images(paths) right b4 return'   
		RTD.report_obj_contents(title,color_images)
		return color_images


# class CameraSettings:
# 	def __init__(self, full_path):

# 		self.exposure = 0
# 		self.fstop = 0
# 		self.iso = 0
# 		self.focal_length = 0
# 		self.bodyserialnumber = []
# 		self.datetimeoriginal = []
# 		self.get_exif_data(full_path)



# 	def get_exif_data(self, full_path):
# 		im = open(full_path,'rb')
# 		tags = exifread.process_file(im)
# 		for tag in tags.keys():
# 		    if tag in "EXIF ExposureTime":
# 		        result = str(tags[tag])
# 		        if '/' in result:
# 		            new_result, x = result.split('/')
# 		            self.exposure = int(new_result)/int(x)
# 		        else:
# 		            self.exposure = int(result)    

# 		    if tag in "EXIF FNumber":
# 		        result = str(tags[tag])
# 		        if '/' in result:
# 		            new_result, x = result.split('/')
# 		            self.fstop = int(new_result)/int(x)
# 		        else:
# 		            self.fstop = int(result)    

# 		    if tag in "EXIF ISOSpeedRatings":
# 		        result = str(tags[tag])
# 		        self.iso = int(result)

# 		    if tag in 'EXIF DateTimeOriginal':
# 		        result = str(tags[tag])
# 		        self.datetimeoriginal = result
# 		        #print(self.datetimeoriginal)

# 		    if tag in "EXIF BodySerialNumber":
# 		        result = str(tags[tag])
# 		        self.bodyserialnumber = result
# 		        #print(self.bodyserialnumber)

# 		    if tag in "EXIF FocalLength":
# 		        result = str(tags[tag])
# 		        self.focal_length = result
# 		        #print(self.focal_length)




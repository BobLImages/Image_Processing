# color_image.py

from image_functions import ImageFunctions as IF
import cv2
import numpy as np


class ColorImage:
	def __init__(self, r_src, image_id=None, fname=None, geometry=None):
		self.image_id = image_id
		self.image_path = fname
		self.ro_image = r_src

		# Handle geometry cleanly
		self.geometry = geometry 

		# Preprocessing
		img_dnz = cv2.fastNlMeansDenoisingColored(self.ro_image.astype(np.uint8), None, 2, 2, 7, 21)
		self.RODS_image = IF.sharpen(img_dnz)
		self.image_gs = cv2.cvtColor(self.RODS_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)


		        # === Add core stats right here ===
		self.brightness = IF.get_brightness(self.RODS_image)
		self.contrast = IF.get_contrast(self.image_gs)
		self.laplacian = IF.get_laplacian(self.image_gs)






		# cv2.imshow("Debug Image", self.RODS_image)
		# cv2.waitKey(0)  # Wait for any key
		# cv2.destroyAllWindows()


		# Stats and camera
		# self.stats = ImageStatistics(self.image_sharp, self.image_gs)
		# self.camera = CameraSettings(self.image_path)
		# self.get_classification()
		print(f"Processing Index: {self.image_id}  File: {self.image_path.name}")


# class  ImageStatistics:
# 	def __init__(self, image_sharp,image_gs):

# 		self.brightness = get_brightness(image_sharp)

# 		self.contrast = get_contrast(image_gs)

# 		self.haze_factor = get_haze_factor(image_gs)

# 		self.hough_lines = get_hough_lines(image_gs)

# 		self.hough_circles = get_hough_circles(image_gs)    

# 		self.harris_corners = get_harris(image_gs)

# 		self.contour_info  = get_contours(image_gs)

# 		self.laplacian = get_laplacian(image_gs)

# 		self.variance = get_variance(image_gs)

# 		self.shv = get_shv(image_gs)

# 		self.faces = get_faces(image_gs)

# 		self.eyes = get_eyes(image_gs)

# 		self.bodies = get_bodies(image_gs)


class CameraSettings:
	def __init__(self, full_path):

		self.exposure = 0
		self.fstop = 0
		self.iso = 0
		self.focal_length = 0
		self.bodyserialnumber = []
		self.datetimeoriginal = []
		self.get_exif_data(full_path)



	def get_exif_data(self, full_path):
		im = open(full_path,'rb')
		tags = exifread.process_file(im)
		for tag in tags.keys():
		    if tag in "EXIF ExposureTime":
		        result = str(tags[tag])
		        if '/' in result:
		            new_result, x = result.split('/')
		            self.exposure = int(new_result)/int(x)
		        else:
		            self.exposure = int(result)    

		    if tag in "EXIF FNumber":
		        result = str(tags[tag])
		        if '/' in result:
		            new_result, x = result.split('/')
		            self.fstop = int(new_result)/int(x)
		        else:
		            self.fstop = int(result)    

		    if tag in "EXIF ISOSpeedRatings":
		        result = str(tags[tag])
		        self.iso = int(result)

		    if tag in 'EXIF DateTimeOriginal':
		        result = str(tags[tag])
		        self.datetimeoriginal = result
		        #print(self.datetimeoriginal)

		    if tag in "EXIF BodySerialNumber":
		        result = str(tags[tag])
		        self.bodyserialnumber = result
		        #print(self.bodyserialnumber)

		    if tag in "EXIF FocalLength":
		        result = str(tags[tag])
		        self.focal_length = result
		        #print(self.focal_length)




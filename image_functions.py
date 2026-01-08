# image_functions.py

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
#from skimage.morphology import rectangular
# from skimage.morphology import rectangle
# from skimage.filters.rank import mean as rank_mean
import mediapipe as mp

class ImageFunctions:
    _face_detection = None  # class-level cache so we don't recreate it every call

    @staticmethod
    def detect_faces_with_mediapipe(img_cv, min_confidence=0.5):
        """Detect faces in an OpenCV BGR image → return list of face dicts"""
        import mediapipe as mp
        import cv2

        if img_cv is None:
            return []

        # Create detector fresh each call — simple and reliable
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,          # 1 = full range (better for group shots)
            min_detection_confidence=min_confidence
        )

        # Convert BGR → RGB
        rgb_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        faces = []
        if results.detections:
            h, w = img_cv.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                confidence = detection.score[0]

                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(confidence),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })

        return faces

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
    def resize_keep_aspect(image, max_width=600, max_height=800):
        """
        Resize an image to fit within max_width x max_height while keeping aspect ratio.
        """
        if image is None:
            raise ValueError("resize_keep_aspect received None image")

        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def sharpen(image, kernel_strength = 1):
        """
        Sharpen an image using a simple kernel.
        """
        if image is None:
            raise ValueError("sharpen received None image")

        if kernel_strength == 1:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
        elif kernel_strength== 2:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]]) / 9.0   # normalized Laplacian, less aggressive
        
        elif kernel_strength== 3:
            kernel = np.array([[-1, -1, -1],
                               [-1,  6, -1],
                               [-1, -1, -1]]) / 2.0   # half strength
        

        elif kernel_strength == 4:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])/4.0

        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def grayscale(image):
        """
        Convert a BGR image to grayscale.
        """
        if image is None:
            raise ValueError("grayscale received None image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def denoise(image, h_color=2, h_luminance=2, template_window=7, search_window=21):
        """
        Denoise a color image using OpenCV's fastNlMeansDenoisingColored.
        Parameters are tuned for mild denoising.
        """
        if image is None:
            raise ValueError("denoise received None image")

        # Make sure image is uint8
        img_uint8 = image.astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h_color, h_luminance, template_window, search_window
        )
        return denoised





    @staticmethod
    def get_laplacian(image_gs):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        
        return cv2.Laplacian(image_gs, cv2.CV_64F,5).var()


    @staticmethod
    def get_harris(image_gs):
        gray = np.float32(image_gs)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        return len(corners)

    @staticmethod
    def get_brightness(image_sharp):
        if len(image_sharp.shape) == 3:
            hsv = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2HSV)
            v_channel = hsv[..., 2]
            brightness = v_channel.mean()
            # print(f"Brightness for ImageStats: {brightness:.2f} (min: {v_channel.min()}, max: {v_channel.max()})")
        else:
            brightness = 0
            # print(f"Brightness fallback (grayscale image): {brightness}")

        return brightness
        
    @staticmethod
    def get_contrast(image_gs):
        std_dev = image_gs.std()
        min_val = image_gs.min()
        max_val = image_gs.max()
        if std_dev < 20:  # Too flat - tweakable
            contrast = std_dev
            status = "Too flat"
        elif std_dev > 60 and (min_val < 5 or max_val > 250):  # Too harsh, tied to extremes
            contrast = std_dev
            status = "Too harsh"
        else:
            contrast = std_dev
            status = "Good"
        # print(f"Contrast: {contrast:.2f} ({status}, min: {min_val}, max: {max_val})")
        return contrast

    @staticmethod
    def get_haze_factor(image_sharp):
        if len(image_sharp.shape) == 3:
            hsv = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2HSV)
            s_channel = hsv[..., 1]  # Saturation (0-255)
            saturation = s_channel.mean()
            brightness = hsv[..., 2].mean()
            # Hazy: moderate brightness, low saturation
            if saturation < 50 and 50 < brightness < 150:
                haze_factor = 100 - saturation  # Higher = hazier
                status = "Hazy"
            else:
                haze_factor = saturation
                status = "Clear"
            # print(f"Haze: {haze_factor:.2f} ({status}, sat: {saturation:.2f}, bright: {brightness:.2f})")
        else:
            haze_factor = 0
            # print(f"Haze fallback: {haze_factor}")
        return haze_factor


    @staticmethod
    def get_variance(image_gs):

        # img = image_gs.astype(np.uint8)
        # img_sq = np.square(img).astype(np.uint16)

        # footprint = rectangular(5, 5)

        # mean_img = rank_mean(img, footprint)
        # mean_img_sq = rank_mean(img_sq, footprint)

        # variance = mean_img_sq.astype(np.float64) - np.square(mean_img.astype(np.float64))
        global_variance =  2.2 # np.mean(variance)
        # print(f'Global Variance {global_variance}')
        return float(global_variance)

    @staticmethod
    def get_shv(image_gs):

        # print(f'shape: {image_gs.shape}')
        width, height = image_gs.shape
        pix = image_gs

        vs = []
        
        # print(f'height: {height}')
        # print(f'width: {width}')
        for y in range(height):
            row = [pix[x,y] for x in range(width)]
            # int(mean) = sum(row)/width
            # variance = sum([(x-mean)**2 for x in row])/width

            mean = np.sum(np.array(row, dtype=np.float64)) / width
            variance = np.sum(np.array([(x - mean) ** 2 for x in row], dtype=np.float64)) / width
            vs.append(variance)
        return np.sum(np.array(vs, dtype=np.float64)) / height



class EXIFFunctions:

    @staticmethod
    def get_exposure(result):
        if '/' in result:
            new_result, x = result.split('/')
            exposure = int(new_result)/int(x)
        else:
            exposure = int(result)    
        return exposure

    @staticmethod
    def get_f_stop(result):
        if '/' in result:
            new_result, x = result.split('/')
            fstop = int(new_result)/int(x)
        else:
            fstop = int(result)    
        return fstop
     
    @staticmethod
    def get_iso(result):
        iso = int(result)
        return

    @staticmethod
    def get_focal_length(result):
        focal_length = int(result)
        return focal_length














































# import time
# from PIL import Image, ImageStat
# import cv2 as cv2
# import numpy as np
# import math
# import sys
# import os
# # import matplotlib.pyplot as plt
# # from skimage.morphology import rectangle
# # import skimage.filters as filters
# # from skimage import img_as_ubyte

# # from skimage import io,img_as_float,transform
# # from skimage.restoration import denoise_nl_means,estimate_sigma
# # from skimage.util import img_as_ubyte
# # from scipy import ndimage
# # from scipy.ndimage.filters import convolve
# from numpy.linalg import norm




# def get_hough_lines(image_gs):

#     line_length = 800
#     edges = cv2.Canny(image_gs, 125, 350, apertureSize=3)
#     success = False
#     while not success:
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, line_length)
#         try:
#             if len(lines) > 0:
#                 #print(self.hough_info)
#                 success = True
#                 return [len(lines),line_length]

#         except:
#             line_length = line_length -100
#             if line_length < 20:
#                 success = True
#                 return [0,20]


# def get_hough_circles(image_gs):
#     gray = cv2.GaussianBlur(image_gs, (5, 5), 0)  # Add blur
#     circles = cv2.HoughCircles(
#         gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
#         param1=100, param2=20, minRadius=10, maxRadius=200
#     )
#     print(f"Circles detected: {self.hough_circles}")
#     return len(circles[0]) if circles is not None else 0





# def get_contours(image_gs):

#     edged = cv2.Canny(image_gs, 50, 150, 3)
#     contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     try:
#        return len(contours),0
#     except:
#         return (0,0)



# def get_faces(image_gs):
#     # Correct path with (x86) and Windows slashes
#     cascade_path = "C:/Program Files (x86)/Sublime Text/haarcascade_frontalface_default.xml"
#                   # "C:\\Program Files (x86)\\Sublime Text\\haarcascade_frontalface_default.xml"        
#     # Load the classifier
#     face_cascade = cv2.CascadeClassifier(cascade_path)
    
#     # Check if loaded
#     if face_cascade.empty():
#         print(f"Error: Couldn’t load cascade from {cascade_path}")
#         return 0

#     # Detect faces
#     faces = face_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
#     return len(faces)     



# def get_eyes(image_gs):
#     # Fix path to your dir
#     cascade_path = r"C:/Program Files (x86)/Sublime Text/haarcascade_eye.xml"
    
#     # Load it
#     eye_cascade = cv2.CascadeClassifier(cascade_path)
    
#     # Safety check
#     if eye_cascade.empty():
#         print(f"Error: Can’t load {cascade_path}—check path or file!")
#         return 0

    
#     # Detect eyes
#     eyes = eye_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
#     return len(eyes)


# def get_bodies(image_gs):

#     # Path to the Haar cascade XML file
#     cascade_path = "C:/Program Files (x86)/Sublime Text/haarcascade_fullbody.xml"
#                  #  "C:/Program Files (x86)/Sublime Text/haarcascadefullbody.xml""
#     # Load the Haar cascade classifier
#     body_cascade = cv2.CascadeClassifier(cascade_path)

#     # Detect bodies in the image
#     bodies = body_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
#     return len(bodies)   



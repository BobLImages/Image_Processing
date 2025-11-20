
# image_catalog.py

from pandas import DataFrame
from typing import Optional, Dict, Any
# from color_image_class import ColorImage # Removed unused import
import cv2
import numpy as np
from PIL import Image
from mask_repository import MaskRepository
from mask_factory import MaskFactory, MASK_DEFS # Ensure MASK_DEFS is imported


class ImageCatalog:
    def __init__(self, images_df: DataFrame):
        self.df = images_df
        self.img_cv: Optional[np.ndarray] = None # Original BGR image
        self.img_gs: Optional[np.ndarray] = None # Grayscale image
        self.img_hsv: Optional[np.ndarray] = None    # HSV image
        self.img_pil = None # Removed, created on demand
        self.repository: Optional[MaskRepository] = None # Renamed to singular as per current structure


    def acquire_image(self,current_index):
        if not (0 <= current_index < len(self.df)):
            print("Index out of bounds")
            return

        print(f'Row: {current_index}')
        row = self.df.iloc[current_index]
        


        img_data = row['Sharpened_Image']
        gs_data = row['Grayscale_Image']

        # --- SAFETY FIRST ---
        if img_data is None or len(img_data) == 0:
            print(f"Missing Sharpened_Image for {row['File_Name']}")
            return
        if gs_data is None or len(gs_data) == 0:
            print(f"Missing Grayscale_Image for {row['File_Name']}")
            return

        self.img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if self.img_cv is None:
            print(f"Failed to decode Sharpened_Image for {row['File_Name']}")
            return

        self.img_gs = cv2.imdecode(np.frombuffer(gs_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.img_gs is None:
            print(f"Failed to decode Grayscale_Image")
            return

        # Now it's safe
        self.img_hsv = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2HSV)
        self.repository = MaskRepository(hsv_shape=self.img_hsv.shape)
        self.pil_img = Image.fromarray(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB))        
        print(f"Acquired image data for '{row['File_Name']}'. Repository initialized.")
                


    



    def get_pure_binary_mask_display(self, mask_name: str) -> Optional[Image.Image]:
        """
        Returns a PIL image of the raw binary mask (white = active, black = inactive).
        """
        mask_array = self.repository.get(mask_name)
        if mask_array is None:
            print(f"Catalog Error: Mask '{mask_name}' not found in repository.")
            return None

        # Create 3-channel image: white where mask > 0
        mask_3ch = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
        mask_3ch[mask_array > 0] = (255, 255, 255)  # White = active

        return Image.fromarray(mask_3ch)


    def get_original_pil_image(self) -> Optional[Image.Image]:
        """
        Returns the original loaded image as a PIL Image (RGB format) for the GUI canvas.
        """
        if self.img_cv is None:
            return None
            
        
        return self.pil_img


    def create_and_store_mask(self, mask_name: str) -> Optional[np.ndarray]:
        """
        Coordinates fetching a stored mask or creating a new one using the Factory
        and storing it in the Repository (implements caching logic).
        """
        if self.repository is None or self.img_hsv is None or self.img_cv is None:
            print("Error: Image data or repository not initialized.")
            return None

        # 1. Check repository cache first
        raw_mask_array = self.repository.get(mask_name)
        if raw_mask_array is not None:
            print(f"Mask '{mask_name}' retrieved from repository cache.")
            return raw_mask_array

        # 2. Get the definition/recipe from the Factory's definitions
        definition = MaskFactory.get_mask_definition(mask_name) # Assuming this function exists in MaskFactory
        if definition is None:
            print(f"Unknown mask name or missing definition: '{mask_name}'.")
            return None
            
        method = definition['method']
        params = definition['params']
        raw_mask_array = None

        if method == 'multi_hsv_range':
            raw_mask_array = MaskFactory.create_multi_hsv_range_mask(self.img_hsv, **params)
        elif method == 'hsv_value_mask':
            raw_mask_array = MaskFactory.create_hsv_value_mask(self.img_hsv, **params)
        elif method == 'grayscale_threshold': # <-- New case
            raw_mask_array = MaskFactory.create_grayscale_threshold_mask(self.img_cv, **params)
        elif method == 'grayscale_adaptive':  # <-- New case
            raw_mask_array = MaskFactory.create_grayscale_adaptive_mask(self.img_cv, **params)
        elif method == 'grayscale_canny':     # <-- New case
            raw_mask_array = MaskFactory.create_grayscale_canny_mask(self.img_cv, **params)
        # elif method == 'kmeans_mask':

        # 4. Store the newly created mask in the repository and return it
        if raw_mask_array is not None:
            self.repository.add(mask_name, raw_mask_array)
            print(f"Generated and stored mask '{mask_name}'.")
            return raw_mask_array
        
        return None


    def generate_blended_mask_for_display(self, mask_name: str, mask_array: np.ndarray) -> Optional[Image.Image]:
        """
        Generates the blended, color-tinted masked image as a PIL Image object,
        ready for the Tkinter canvas. This handles jobs 2, 3, 4, and 5 internally.
        """
        if self.img_cv is None:
            print("Catalog Error: No base image loaded for blending.")
            return None
            
        img_bgr = self.img_cv.copy() # Local copy of the original BGR image

        # 2. Get overlay colour and blend info from the mask definition
        cfg = MASK_DEFS.get(mask_name, {}) # Use MASK_DEFS imported from Factory
        overlay_color = cfg.get('overlay_color', (0, 0, 255))
        blend_alpha = cfg.get('blend_alpha', 0.7) # Use 0.4 as standard
        blend_beta = 1.0 - blend_alpha

        # 3. Build the coloured overlay
        overlay = img_bgr.copy()
        overlay[mask_array > 0] = overlay_color          # colour only the masked pixels

        # 4. Blend (e.g., 60% original + 40% overlay)
        blended = cv2.addWeighted(img_bgr, blend_beta, overlay, blend_alpha, 0)
        
        # 5. Convert to PIL format (The calling function will display it)
        pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        
        return pil_img








# # image_catalog.py
# from pandas import DataFrame
# from typing import Optional, Dict, Any
# from color_image_class import ColorImage 
# import cv2
# import numpy as np
# from PIL import Image, ImageTk, ImageDraw
# from mask_repository import MaskRepository
# from mask_factory import MaskFactory

# class ImageCatalog:
#     def __init__(self, images_df: DataFrame):
#         self.df = images_df
#         self.current_index = 0
#         self.img_cv = None
#         self.img_gs = None
#         self.hsv = None
#         self.img_pil = None
#         self.mask_repository = {}
#         self.factory = {}




#     def ensure_hsv(self):
#         print(f'Inside ensure_hsv')
#         if self.hsv is None:
#             if self.img_cv is not None:
#                 self.hsv = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2HSV)
#                 self.mask_repository =  MaskRepository(self.hsv.shape)  # ← CLEAR: REPOSITORY
#         return self.hsv


#     def ensure_cv2pil(self):
#         img_rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
#         return pil_img

#     def get_image(self, state = 'display') -> Optional[Dict[str, Any]]:
#         if self.df.empty or self.current_index >= len(self.df):
#             return None

#         row = self.df.iloc[self.current_index]
#         self.acquire_image()
        
#         if state == 'display':
#             self.img_pil = self.ensure_cv2pil()
#             img = self.img_pil
#         else:
#             img = self.img_cv     

#         image_dict =  {
#                 'image' : img,
#                 'id': row['Image_ID'],
#                 'filename': row['File_Name'],
#                 'classification': row['Classification'],
#                 # Add more: SNR, date, etc.
#             }
#         return img # age_dict

 

#     def generate_blended_mask_for_display(self, mask_name: str,final_mask_array) -> Optional[Image.Image]:

#         if self.img_cv is None:
#             print("Catalog Error: No base image loaded for blending.")
#             return None
            
#         img_bgr = self.img_cv.copy() # Local copy of the original BGR image

#         # 2. Get overlay colour from the mask definition (Requires MaskFactory import)
#         cfg = MaskFactory.MASK_DEFS.get(mask_name, {})
#         overlay_color = cfg.get('overlay_color', (0, 0, 255))   # default = red

#         # 3. Build the coloured overlay
#         overlay = img_bgr.copy()
#         overlay[final_mask_array > 0] = overlay_color          # colour only the masked pixels

#         # 4. Blend (70 % original + 30 % overlay)
#         blended = cv2.addWeighted(img_bgr, 0.7, overlay, 0.3, 0)
        
#         # 5. Convert to PIL format (The calling function will display it)
#         pil_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        
#         # We return the finished PIL image instead of displaying it ourselves
#         return pil_img



#     def create_and_store_mask(self, mask_name: str) -> Optional[np.ndarray]:
#         # ... (check repository cache first, as before) ...
#         raw_mask_array = self.repository.get(mask_name)
#         if raw_mask_array is not None:
#             return raw_mask_array

#         print(f"Mask '{mask_name}' not found in cache. Generating now...")

#         # 1. Get the definition/recipe from the Factory's definitions
#         definition = MaskFactory.get_mask_definition(mask_name)
#         if definition is None:
#             print(f"Unknown mask name or missing definition: '{mask_name}'.")
#             return None
            
#         method = definition['method']
#         params = definition['params']
#         raw_mask_array = None

#         # 2. Use the correct Factory method based on the 'method' string
#         if method == 'hsv_range':
#             raw_mask_array = MaskFactory.create_hsv_range_mask(self.hsv, **params)
#         elif method == 'hsv_value_mask':
#             # Use 'self.hsv' data
#             raw_mask_array = MaskFactory.create_hsv_value_mask(self.hsv, **params)
#         elif method == 'grayscale_mask':
#             # This method needs the original BGR data 'self.img_cv'
#             raw_mask_array = MaskFactory.create_grayscale_mask(self.img_cv, **params)
#         # Add more elifs here for K-Means later

#         # 3. Store the newly created mask in the repository and return it
#         if raw_mask_array is not None:
#             self.repository.add(mask_name, raw_mask_array)
#             return raw_mask_array
        
#         return None

#     def get_original_pil_image(self) -> Optional[Image.Image]:
#         """
#         Returns the original loaded image (BGR format) converted into 
#         a PIL Image (RGB format) for the GUI canvas.
#         """
#         if self.img_cv is None:
#             return None
            
#         # Convert BGR (OpenCV format) to RGB (PIL format)
#         img_rgb = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
        
#         return pil_img




    


#     # def get_mask(self):
   
#     #     processing_dict = self.get_image('processing')
#     #     self.factory = MaskFactory(catalog=self)


































            
#     def get_mask(self, preset='red'):
#         print(f'Getting Mask')
#         if self.df.empty or self.current_index >= len(self.df):
#             return None
        
#         mask_array = MaskFactory.build(preset, catalog = self)
        





#         for key, value in self.factory.items():
#             print(f"{key}: {value}")
#         self.factory
#         # if mask_array is not None:
#         #     self.basic_mask.add(preset, mask_array)
#         #     self.





#     def acquire_image(self):

#             # Acquire df images
#             row = self.df.iloc[self.current_index]
#             print(f'Row: {row}')
#             img_data = row['Sharpened_Image']
#             np_array = np.frombuffer(img_data, np.uint8)
#             gs_data = row['Grayscale_Image']
#             gs_np_array = np.frombuffer(gs_data, np.uint8)

#             # Decode image
#             self.img_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
#             self.img_gs = cv2.imdecode(gs_np_array, cv2.IMREAD_GRAYSCALE)  # 1 channel
#             self.hsv = self.ensure_hsv()
#             self.img_pil = self.get_original_pil_image
















#         # # # 3. BUILD THE MASK
#         # # mask_array = self.factory.create_preset('blue')  # ← THIS WAS MISSING

#         # # if mask_array is None:
#         # #     return None

#         # # # 4. Store it
#         # # self.masks.add('blue', mask_array)

#         # # 5. Blend and return
#         # cv_img = self.img_cv
#         # overlay = np.zeros_like(cv_img)
#         # overlay[mask_array != 0] = [255, 0, 0]  # Blue
#         # masked_cv = cv2.addWeighted(cv_img, 0.7, overlay, 0.3, 0)

#         # pil_img = Image.fromarray(cv2.cvtColor(masked_cv, cv2.COLOR_BGR2RGB))
#         # return {'image': pil_img, 'filename': self.df.iloc[self.current_index]['File_Name']}

#     def combine_masks(self, names, operation='or'):
#         if not names or not self.masks:
#             return np.zeros_like(self._hsv[:,:,0])
#         result = self.masks[names[0]].copy()
#         for name in names[1:]:
#             if operation == 'or':
#                 result = cv2.bitwise_or(result, self.masks[name])
#             elif operation == 'and':
#                 result = cv2.bitwise_and(result, self.masks[name])
#         return result


#     def get_displayable_masked_image(self, mask_name: str) -> Optional[Image.Image]:
#         """
#         Generates the blended, color-tinted masked image as a PIL Image object,
#         ready for the Tkinter canvas.
#         """
#         if self.img_cv is None:
#             print("Catalog Error: No base image loaded.")
#             return None
        
#         # 1. Get the raw mask array from the repository
#         mask_array = self.repository.get(mask_name)
#         if mask_array is None:
#             print(f"Catalog Error: Mask '{mask_name}' not found in repository.")
#             # We should likely call create_and_store_mask here automatically if we trust the input name
#             return None

#         # 2. Get the overlay configuration (color, blend opacity) from the Factory/Definitions
#         cfg = MaskFactory.MASK_DEFS.get(mask_name, {})
#         overlay_bgr = cfg.get('overlay_color', (0, 0, 255))   # default = red
#         blend_alpha = cfg.get('blend_alpha', 0.3)             # default = 0.3 (30% overlay strength)
#         blend_beta = 1.0 - blend_alpha                        # 70% original strength

#         # 3. Build the coloured overlay image using a copy of the original
#         overlay = self.img_cv.copy()
#         # Colour only the masked pixels using boolean indexing
#         # Note: mask_array > 0 ensures we select all non-zero pixels
#         overlay[mask_array > 0] = overlay_bgr          

#         # 4. Blend (using the alpha/beta from config)
#         # We blend the original image (self.img_cv) with the overlay image
#         blended_bgr = cv2.addWeighted(self.img_cv, blend_beta, overlay, blend_alpha, 0)

#         # 5. Convert the blended BGR image to a PIL Image (RGB format)
#         img_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(img_rgb)
        
#         return pil_img







#     def prepare_image_for_display(self, image_bgr: np.ndarray, target_width: int, target_height: int) -> Optional[Image.Image]:
#         """
#         Resizes a BGR image using OpenCV while maintaining aspect ratio, 
#         and converts it to a PIL Image (RGB) ready for the GUI canvas.
#         """
#         if image_bgr is None:
#             return None
            
#         h, w = image_bgr.shape[:2]
        
#         # Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
#         scale = min(target_width / w, target_height / h)
        
#         if scale >= 1.0:
#             # Image is already smaller than canvas, no need to resize
#             resized_bgr = image_bgr
#         else:
#             new_w = int(w * scale)
#             new_h = int(h * scale)
#             # Use OpenCV's resize (uses W, H order for dsize argument)
#             resized_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
#         # Convert BGR (OpenCV) to RGB (PIL) format
#         img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        
#         # Convert NumPy array to PIL object
#         pil_img = Image.fromarray(img_rgb)
        
#         return pil_img
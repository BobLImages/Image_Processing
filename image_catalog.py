
# image_catalog.py

from pandas import DataFrame
from typing import Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image
from mask_repository import MaskRepository
from mask_factory import MaskFactory
from mask_defs import MASK_DEFS
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ImageRef:
    db_name: str
    image_name: str

    @classmethod
    def from_path(cls, path: Path) -> "ImageRef":
        return cls(
            db_name=path.parent.name,
            image_name=path.stem
        )



class ImageCatalog:
    def __init__(self, images_df: DataFrame):
        self.df = images_df
        self.img_org_cv: Optional[np.ndarray] = None # Original BGR image
        self.img_sharp_cv: Optional[np.ndarray] = None # Sharpened BGR image
        self.img_gs: Optional[np.ndarray] = None # Grayscale image
        self.img_hsv: Optional[np.ndarray] = None    # HSV image
        self.img_pil = None
        self.repository: Optional[MaskRepository] = None # Renamed to singular as per current structure
        self.image_ref = None


    def acquire_image(self,current_index):
        if not (0 <= current_index < len(self.df)):
            print("Index out of bounds")
            return

        print(f'Row: {current_index}')
        row = self.df.iloc[current_index]
                
                
        file_path = row["File_Name"]

        self.image_ref = ImageRef(
            db_name=file_path.parent.name,
            image_name=file_path.stem
        )

        sharpened = row['Sharpened_Image']
        grayscale = row['Grayscale_Image']
        original = row['Original_Image']

        # --- SAFETY FIRST ---
        if sharpened is None or len(sharpened) == 0:
            print(f"Missing Sharpened_Image for {row['File_Name']}")
            return
        if original is None or len(original) == 0:
            print(f"Missing Original_Image for {row['File_Name']}")
            return
        if grayscale is None or len(grayscale) == 0:
            print(f"Missing Grayscale_Image for {row['File_Name']}")
            return

        self.img_org_cv = original
        if self.img_org_cv is None:
            print(f"Failed to decode Original_Image for {row['File_Name']}")
            return
        
        self.img_sharp_cv = sharpened
        if self.img_sharp_cv is None:
            print(f"Failed to decode Sharpened_Image for {row['File_Name']}")
            return

        self.img_gs =  grayscale
        if self.img_gs is None:
            print(f"Failed to decode Grayscale_Image")
            return

        self.img_hsv = cv2.cvtColor(self.img_sharp_cv, cv2.COLOR_BGR2HSV)
        self.repository = MaskRepository(hsv_shape=self.img_hsv.shape)
        self.pil_img = Image.fromarray(cv2.cvtColor(self.img_sharp_cv, cv2.COLOR_BGR2RGB))        
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
        if self.img_sharp_cv is None:
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
        if self.img_sharp_cv is None:
            print("Catalog Error: No base image loaded for blending.")
            return None
            
        img_bgr = self.img_sharp_cv.copy() # Local copy of the original BGR image

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


    def report(self, rows: int = 10):
        """Print a nice summary of the current catalog DF."""
        if self.df is None or self.df.empty:
            print("Catalog is empty.")
            return

        print(f"\n=== Image Catalog Report ===")
        print(f"Total images: {len(self.df)}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Current index: {self.current_index}")
        print(f"Current file: {self.df.iloc[self.current_index]['File_Name'] if 0 <= self.current_index < len(self.df) else 'None'}")

        print(f"\n--- First {rows} rows ---")
        print(self.df.head(rows).to_string(index=False))

        # Optional: show stats summary
        if 'Brightness' in self.df.columns:
            print(f"\nBrightness stats:")
            print(self.df['Brightness'].describe())

        if 'Laplacian_Var' in self.df.columns:
            print(f"\nFocus (Laplacian Variance) stats:")
            print(self.df['Laplacian_Var'].describe())

        print(f"\n=== End Report ===\n")

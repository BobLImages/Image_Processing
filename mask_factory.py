# mask_factory.py

import cv2
import numpy as np
from image_functions import ImageFunctions as IF
from PIL import Image, ImageTk, ImageDraw
from mask_defs import MASK_DEFS
from rpt_dbg_tst import RTD


class MaskFactory:

    @staticmethod
    def build(mask_name: str, four_amigos:dict, repository):
        print(f'Inside build() - Creating mask: {mask_name}')

        config = MASK_DEFS.get(mask_name)
        if config is None:
            print(f"Error: Mask preset '{mask_name}' not found.")
            return None
        base_image = four_amigos.get("original")
        final_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)

        mask_type = config['type']
        print(f"The type for '{mask_name}' is: {mask_type}")

        # Use 'hsv' for color, anything else goes to the bw builder (including 'g_s')
        if mask_type == 'hsv':
            hsv_image = four_amigos.get('hsv')
            final_mask = MaskFactory._color_builder(config, mask_name, hsv_image)
        elif mask_type == 'demolition':
            original_image = four_amigos.get('original')
            MaskFactory._demolition_builder(config, mask_name, original_image)
        elif mask_type == 'demolition_slides':
            pass
        elif mask_type == 'sharpen_delta':
            base = four_amigos.get('original')
            overlay = four_amigos.get('sharpened')
            final_mask = MaskFactory._delta_builder(config, mask_name,base, overlay )
        else:
            gs_image = four_amigos.get('greyscale')
            final_mask = MaskFactory._bw_builder(config, mask_name, gs_image)

        if final_mask is None:
            print("No mask created or mask build failed.")
            return None

        # === 3. STORE & SHOW (Assumes catalog structure is correct) ===
        repository.masks[mask_name] = final_mask 
        return final_mask
        # self.display_mask_test_windows(mask_name, final_mask_array)

    @staticmethod
    def _delta_builder(config, mask_name, base, overlay):
        amplify = config.get('amplify', 5)

        # Get the two images (from catalog attributes or DF row)
        img1 = base
        img2 = overlay
        if img1 is None or img2 is None:
            return np.zeros(base.shape[:2], dtype=np.uint8)

        delta = cv2.absdiff(img1.astype(np.float32), img2.astype(np.float32))
        delta_sum = np.sum(delta, axis=2) if len(delta.shape) == 3 else delta
        delta_amp = delta_sum * amplify
        delta_mask = np.clip(delta_amp, 0, 255).astype(np.uint8)

        return delta_mask



    @staticmethod
    def _color_builder(config, mask_name, hsv_image):
        if config is None or config['type'] != 'hsv':
            return None
        
        # Ensure we have the base mask array ready, using the shape of the HSV image
        final_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)        

        # --- Check if we use 'ranges' (e.g., 'red', 'green') ---
        if 'ranges' in config:
            # Iterate through the list of (lower, upper) tuples
            for lower, upper in config['ranges']:
                binary_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                final_mask = cv2.bitwise_or(final_mask, binary_mask)

        # --- Check if we use 'v_min'/'v_max' (e.g., 'dark', 'mid') ---
        elif 'v_min' in config and 'v_max' in config:
            v_min = config['v_min']
            v_max = config['v_max']
            # Define single range for V-channel only, covering all H and S
            lower_bound = np.array([0, 0, v_min])
            upper_bound = np.array([179, 255, v_max])
            final_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        else:
            print(f"Error: Preset '{mask_name}' has an unsupported HSV configuration.")
            return None

        return final_mask


    @staticmethod
    def _demolition_builder(config, mask_name, original_image):
        demos = []
        # Retrieve the specific config dict (e.g., the 'red' or 'mid' dict)
        if config['type'] != 'demolition':
            return None

        """Build a full-color demolished image by repeated sharpening."""
        iterations = config.get('iterations', 0)
        
        if iterations == 0 or original_image is None:
            # Fallback to current displayed image
            return catalog.pil_img
        
        current = original_image.copy()
        for _ in range(iterations):
            current = IF.sharpen(current,4)
            demos.append(current)
        
        RTD.test_slideshow(demos)


    @staticmethod
    def _bw_builder(config, mask_name, gs_image):
        """
        Build any grayscale mask using ONLY the config dict.
        No defaults. No magic numbers.
        """
        if gs_image is None:
            print(f"Error: No grayscale image for '{mask_name}'")
            return None

        mask_type = config.get('type')
        if mask_type == 'g_s':  # Grayscale-style
            sub_type = config.get('sub_type') or mask_name

            if sub_type == 'canny':
                low = config['low']
                high = config['high']
                aperture = config.get('aperture', 3)
                l2grad = config.get('l2gradient', False)
                return cv2.Canny(gs_image, low, high, apertureSize=aperture, L2gradient=l2grad)

            elif sub_type in ['threshold', 'gray25_75']:
                thresh = config['thresh'] if 'thresh' in config else config['low']
                maxval = config['maxval'] if 'maxval' in config else config['high']
                thresh_type = config.get('thresh_type', cv2.THRESH_BINARY)
                _, binary = cv2.threshold(gs_image, thresh, maxval, thresh_type)
                return binary


            elif sub_type == 'adaptive':
                method = config.get('method', cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                block = config.get('block_size', 11)
                C = config.get('C', 2)

                # block_size must be odd and > 1
                if block % 2 == 0:
                    block += 1
                if block < 3:
                    block = 3

                return cv2.adaptiveThreshold(
                    gs, 255, method, cv2.THRESH_BINARY,
                    blockSize=block, C=C
                )

            else:
                print(f"Unknown g_s sub_type: {sub_type}")
                return None

        else:
            print(f"Unknown mask type in _build_bw: {mask_type}")
            return None



    # The __init__ just stores the catalog reference if you instantiate the class
    def __init__(self, catalog=None):
        self.catalog = catalog  # to add to repository (optional use)


    def create_preset(self, name):
        ranges = self.PRESETS.get(name)
        if not ranges:
            return None
        return self._create_from_ranges(name, ranges)

    def create_custom(self, name, ranges):
        """ranges = list of (lower, upper)"""
        return self._create_from_ranges(name, ranges)

    def _create_from_ranges(self, name, ranges):
        mask_array = np.zeros(self.catalog.hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            m = cv2.inRange(self.catalog.hsv, np.array(lower), np.array(upper))
            mask_array = cv2.bitwise_or(mask_array, m)

        # AUTO-ADD TO REPOSITORY
        metadata = {
            'id': self.catalog.df.iloc[self.catalog.current_index]['Image_ID'],
            'filename': self.catalog.df.iloc[self.catalog.current_index]['File_Name'],
            'classification': self.catalog.df.iloc[self.catalog.current_index]['Classification']
        }
        # self.catalog.factory = .add(name, mask_array, metadata)
        return mask_array


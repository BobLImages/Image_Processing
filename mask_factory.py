# mask_factory.py

import cv2
import numpy as np


MASK_DEFS = {
    # ──────────────────────  HSV colour masks  ──────────────────────
    'orange': {
        'type': 'hsv',
        'ranges': [((15, 100, 100), (25, 255, 255))],
        'overlay_color': (0, 165, 255)
    },

    'black': {
        'type': 'g_s',
        'sub_type': 'threshold',
        'thresh': 128,
        'maxval': 128,           # ≤ 128 → white in mask
        'thresh_type': cv2.THRESH_BINARY_INV,  # INVERT: dark → white
        'overlay_color': (0, 0, 0)  # BGR → black
    },
    'white': {
        'type': 'g_s',
        'sub_type': 'threshold',
        'thresh': 128,
        'maxval': 255,           # > 128 → white in mask
        'thresh_type': cv2.THRESH_BINARY,
        'overlay_color': (255, 255, 255)  # BGR → white
    },











    'red': {
        'type': 'hsv',
        'ranges': [((0, 50, 50), (10, 255, 255)),
                   ((170, 50, 50), (179, 255, 255))],
        'overlay_color': (0, 0, 255)          # BGR → red
    },
    'green': {
        'type': 'hsv',
        'ranges': [((40, 50, 50), (80, 255, 255))],
        'overlay_color': (0, 255, 0)          # BGR → green
    },
    'blue': {                                 # ← NEW
        'type': 'hsv',
        'ranges': [((100, 50, 50), (140, 255, 255))],
        'overlay_color': (255, 0, 0)          # BGR → blue
    },
    'sky': {
        'type': 'hsv',
        'ranges': [((100, 30, 30), (130, 255, 255))],
        'overlay_color': (255, 170, 0)        # BGR → sky-blue
    },
    'skin': {
        'type': 'hsv',
        'ranges': [((0, 20, 70), (20, 255, 255))],
        'overlay_color': (0, 140, 255)        # BGR → flesh-tone
    },
    'darkskin': {
        'type': 'hsv',
        'ranges': [((0, 20, 30), (20, 230, 255))],
        'overlay_color': (0, 140, 255)        # BGR → flesh-tone
    },
    'grass': {
        'type': 'hsv',
        'ranges': [((35, 50, 50), (85, 255, 255))],
        'overlay_color': (0, 200, 0)          # BGR → grass-green
    },

    # ──────────────────────  Value-only masks  ──────────────────────
    'dark':   {'type': 'hsv', 'v_min': 0,   'v_max': 85,
               'overlay_color': (50, 50, 50)},   # dark gray
    'mid':    {'type': 'hsv', 'v_min': 86,  'v_max': 170,
               'overlay_color': (180, 180, 180)},# mid gray
    'bright': {'type': 'hsv', 'v_min': 171, 'v_max': 255,
               'overlay_color': (255, 255, 255)},# white

    # ──────────────────────  Grayscale masks  ──────────────────────
    'canny': {
        'type': 'g_s',
        'sub_type': 'canny',
        'low': 50, 'high': 150,
        'overlay_color': (0, 255, 255)        # BGR → yellow
    },
    'threshold': {
        'type': 'g_s',
        'sub_type': 'threshold',
        'thresh': 127, 'maxval': 255,
        'overlay_color': (255, 255, 0)        # BGR → cyan
    },
    'gray25_75': {
        'type': 'g_s',
        'sub_type': 'threshold',
        'low': 64, 'high': 191,
        'overlay_color': (255, 200, 0)        # BGR → orange
    },

    'adaptive_gaussian': {
        'type': 'g_s',
        'sub_type': 'adaptive',
        'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # weighted local avg
        'block_size': 11,        # 11×11 pixel neighborhood
        'C': 2,                  # subtract 2 from local mean
        'overlay_color': (180, 0, 255)   # BGR → magenta
    },
    'adaptive_mean': {
        'type': 'g_s',
        'sub_type': 'adaptive',
        'method': cv2.ADAPTIVE_THRESH_MEAN_C,      # simple local avg
        'block_size': 15,
        'C': 3,
        'overlay_color': (0, 255, 255)   # BGR → yellow
    },    }





class MaskFactory:




    # MASK_DEFS = {
    #     # === HSV MASKS ===
    #     'red':     {'type': 'hsv', 'ranges': [((0,50,50),(10,255,255)), ((170,50,50),(179,255,255))]},
    #     'green':   {'type': 'hsv', 'ranges': [((40,50,50),(80,255,255))]},
    #     'blue':    {'type': 'hsv', 'ranges': [((100,50,50),(140,255,255))]},
    #     'skin':    {'type': 'hsv', 'ranges': [((0,20,70),(20,255,255))]},
    #     'sky':     {'type': 'hsv', 'ranges': [((100,30,30),(130,255,255))]},
    #     'grass':   {'type': 'hsv', 'ranges': [((35,50,50),(85,255,255))]},

    #     'dark':   {'type': 'hsv', 'v_min': 0,   'v_max': 85},     # V: 0–85
    #     'mid':    {'type': 'hsv', 'v_min': 86,  'v_max': 170},    # V: 86–170
    #     'bright': {'type': 'hsv', 'v_min': 171, 'v_max': 255},    # V: 171–255
        
    #     # === GRAYSCALE MASKS ===
    #     'canny':   {'type': 'g_s', 'low': 50, 'high': 150},
    #     'threshold':  {'type': 'g_s', 'thresh': 127, 'maxval': 255},
    #     'gray25_75': {'type': 'g_s', 'low': 64, 'high': 191},  # 25%–75% of 255
    # }

    @staticmethod
    def build(mask_name, catalog):
        print(f'Inside build() - Creating mask: {mask_name}')

        config = MASK_DEFS.get(mask_name)
        if config is None:
            print(f"Error: Mask preset '{mask_name}' not found.")
            return None

        final_mask = np.zeros(catalog.img_hsv.shape[:2], dtype=np.uint8)

        mask_type = config['type']
        print(f"The type for '{mask_name}' is: {mask_type}")

        # Use 'hsv' for color, anything else goes to the bw builder (including 'g_s')
        if mask_type == 'hsv':
            final_mask = MaskFactory._build_color(mask_name, catalog)
        else:
            final_mask = MaskFactory._build_bw(mask_name, config, catalog)

        if final_mask is None:
            print("No mask created or mask build failed.")
            return None

        # === 3. STORE & SHOW (Assumes catalog structure is correct) ===
        catalog.repository.masks[mask_name] = final_mask 
        return final_mask
        # self.display_mask_test_windows(mask_name, final_mask_array)




    @staticmethod
    def _build_color(mask_name, catalog):
        # Retrieve the specific config dict (e.g., the 'red' or 'mid' dict)
        config = MASK_DEFS.get(mask_name)
        if config is None or config['type'] != 'hsv':
            return None
        
        # Ensure we have the base mask array ready, using the shape of the HSV image
        final_mask = np.zeros(catalog.img_hsv.shape[:2], dtype=np.uint8)

        # --- Check if we use 'ranges' (e.g., 'red', 'green') ---
        if 'ranges' in config:
            # Iterate through the list of (lower, upper) tuples
            for lower, upper in config['ranges']:
                m = cv2.inRange(catalog.img_hsv, np.array(lower), np.array(upper))
                final_mask = cv2.bitwise_or(final_mask, m)

        # --- Check if we use 'v_min'/'v_max' (e.g., 'dark', 'mid') ---
        elif 'v_min' in config and 'v_max' in config:
            v_min = config['v_min']
            v_max = config['v_max']
            # Define single range for V-channel only, covering all H and S
            lower_bound = np.array([0, 0, v_min])
            upper_bound = np.array([179, 255, v_max])
            final_mask = cv2.inRange(catalog.img_hsv, lower_bound, upper_bound)
            
        else:
            print(f"Error: Preset '{mask_name}' has an unsupported HSV configuration.")
            return None

        return final_mask


    @staticmethod
    def _build_bw(mask_name, config, catalog):
        """
        Build any grayscale mask using ONLY the config dict.
        No defaults. No magic numbers.
        """
        if catalog.img_gs is None:
            print(f"Error: No grayscale image for '{mask_name}'")
            return None

        gs = catalog.img_gs

        mask_type = config.get('type')

        if mask_type == 'g_s':  # Grayscale-style
            sub_type = config.get('sub_type') or mask_name

            if sub_type == 'canny':
                low = config['low']
                high = config['high']
                aperture = config.get('aperture', 3)
                l2grad = config.get('l2gradient', False)
                return cv2.Canny(gs, low, high, apertureSize=aperture, L2gradient=l2grad)

            elif sub_type in ['threshold', 'gray25_75']:
                thresh = config['thresh'] if 'thresh' in config else config['low']
                maxval = config['maxval'] if 'maxval' in config else config['high']
                thresh_type = config.get('thresh_type', cv2.THRESH_BINARY)
                _, binary = cv2.threshold(gs, thresh, maxval, thresh_type)
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


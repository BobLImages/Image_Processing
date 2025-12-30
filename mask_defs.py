# mask_defs.py

import numpy as np
import cv2



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
    # (20, 230, 255)
    'darkskin': {
        'type': 'hsv',
        'ranges': [((0, 20, 30), (18, 180, 220))],
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
    },    

    'demolition_normal': {
        'type': 'demolition',
        'iterations': 0
    },
    'demolition_light': {
        'type': 'demolition',
        'iterations': 2
    },
    'demolition_crispy': {
        'type': 'demolition',
        'iterations': 25
    },
    'demolition_halo': {
        'type': 'demolition',
        'iterations': 100
    },
    'demolition_crunchy': {
        'type': 'demolition',
        'iterations': 500
    },
    'demolition_overcooked': {
        'type': 'demolition',
        'iterations': 2000
    },
    'demolition_chaos': {
        'type': 'demolition',
        'iterations': 10000
    },
    'demolition_apocalypse': {
        'type': 'demolition',
        'iterations': 50000
    },

    'demolition_slideshow': {
        'type': 'demolition_slideshow',
        'delay': 0.6
    },

    'difference_generic': {
        'type': 'sharpen_delta',
        'source1': 'original',      # key in catalog or DF column
        'source2': 'sharpened',
        'amplify': 5,
        'overlay_color': (255, 255, 0)  # yellow for changes
    },


}


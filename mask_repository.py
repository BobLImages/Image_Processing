
# # --- Usage Example ---

# # # Let's create a synthetic image for demonstration
# # color_image = np.zeros((image_size, image_size, 3), np.uint8)
# # cv2.rectangle(color_image, (50, 50), (200, 200), (0, 0, 255), -1)  # Red (BGR)
# # cv2.rectangle(color_image, (300, 300), (450, 450), (0, 0, 100), -1) # Dark Red (BGR)
# # cv2.rectangle(color_image, (300, 50), (450, 200), (255, 0, 0), -1)   # Blue (BGR)


# mask_repository.py (The correct, clean version)

import cv2
import numpy as np
from typing import Optional, Dict

class MaskRepository:
    """
    Stores actual NumPy mask arrays (uint8) that match the specific dimensions 
    of the currently loaded image in the ImageCatalog.
    """
    def __init__(self, hsv_shape: tuple):
        self.masks: Dict[str, np.ndarray] = {}
        self.mask_shape = hsv_shape[:2]

    def add(self, name: str, mask_array: np.ndarray):
        """Adds a new mask array to the repository, validating shape."""
        if mask_array.shape == self.mask_shape:
            self.masks[name] = mask_array
            print(f"Added mask '{name}' to repository.")
        else:
            print(f"Warning: Mask shape {mask_array.shape} does not match repository shape {self.mask_shape}. Not added.")

    def get(self, name) -> Optional[np.ndarray]:
        """Returns a stored mask array by name."""
        return self.masks.get(name)

    def list_names(self):
        """Returns a list of all stored mask names."""
        return list(self.masks.keys())
    
    def clear(self):
        """Clears all stored masks (useful when loading a new image)."""
        self.masks = {}

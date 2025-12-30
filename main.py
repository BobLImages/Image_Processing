# main.py — THE ONE TRUE VERSION


import tkinter as tk
from ipa_controller import ImageAppController




# Optional: only if running from outside the project folder
# import sys
# sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAppController(root)
    root.mainloop()



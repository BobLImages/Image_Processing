# rpt_dbg_tst.py

import numpy as np
import cv2
import pandas as pd
from pathlib import Path

'''Here's  entire module. We need Class: RTD and make those methods 
Like ImageFunctions.  Something else I noticed Don't report and 
report_db_contents look eerily similar ?  
'''



class RTD:

    def report_db_contents(db_path: Path, rows: int = 10):
        


        """Print key columns from the saved DB."""
        if not db_path.exists():
            print(f"DB not found: {db_path}")
            return

        try:
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT Image_ID, File_Name, Brightness, Contrast, Laplacian
                    FROM images
                    LIMIT 10
                """
                df = pd.read_sql(query, conn, params=(rows,))
                print(f"\n=== DB CONTENTS ({db_path.name}) ===")
                print(df.to_string(index=False))
                print(f"\nNumeric summary:")
                print(df[['Brightness', 'Contrast', 'Laplacian']].describe().round(2))
                print("=== END DB REPORT ===\n")
        except Exception as e:
            print(f"DB report failed: {e}")



    def report(images_pd, rows: int = 10):
        """Print a nice summary of the current catalog DF."""
        current_index = 0
        if images_pd is None or images_pd.empty:
            print("Catalog is empty.")
            return

        print(f"\n=== Image Catalog Report ===")
        print(f"Total images: {len(images_pd)}")
        print(f"Columns: {list(images_pd.columns)}")
        print(f"Current index: {current_index}")
        print(f"Current file: {images_pd.iloc[current_index]['File_Name'] if 0 <= current_index < len(images_pd) else 'None'}")

        print(f"\n--- First {rows} rows ---")
        print(images_pd.head(rows).to_string(index=False))

        # Optional: show stats summary
        if 'Brightness' in images_pd.columns:
            print(f"\nBrightness stats:")
            print(images_pd['Brightness'].describe())

        if 'Laplacian_Var' in images_pd.columns:
            print(f"\nFocus (Laplacian Variance) stats:")
            print(images_pd['Laplacian'].describe())

        print(f"\n=== End Report ===\n")


    def test_slideshow(color_images, delay=.5):
        for img in color_images:
            frame = img   # or img.image depending on your final name
            if frame is None:
                print("❌ Missing frame in:", img.path)
                continue

            cv2.imshow("Slideshow Test", frame)
            key = cv2.waitKey(int(delay * 1000))
            if key == 27:  # ESC to quit early
                break
        cv2.destroyAllWindows()



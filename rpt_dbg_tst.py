# rpt_dbg_tst.py report_df_contents

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import random



'''Here's  entire module. We need Class: RTD and make those methods 
Like ImageFunctions.  Something else I noticed Don't report and 
report_db_contents look eerily similar ?  
'''



class RTD:

            #Samplle for  report_obj_, df_contents
            # title = f'\nIn' <function name>'
            # RTD.report_obj_contents(title, objs or df)

    def report_obj_contents(title,objs):
        print(title)
        print("=== RICH obj CHECK ===")
        print("Number of objects:", len(objs))
        # print("Columns:", df.columns.tolist())
        j = random.randint(0, 4)
        if len(objs) > j:
            obj = objs[j]
            print(title)
            print(j, " row image_id:", obj.image_id)
            print(j, " row file_path:", obj.image_path)
            print(j, " row geometry.height:", obj.geometry.get("original_height"))
            print(j, " row ro_image type:", type(obj.ro_image))
            print(j, " row image_stats.brightness:", obj.image_stats.brightness)
            while True:
                k = random.randint(0, 4)
                if k != j:
                    break
            j=k
            obj = objs[j]
            print(f'Second test row')
            print(title)
            print(j, " row image_id:", obj.image_id)
            print(j, " row file_path:", obj.image_path)
            print(j, " row geometry.height:", obj.geometry.get("original_height"))
            print(j, " row ro_image type:", type(obj.ro_image))
            print(j, " row image_stats.brightness:", obj.image_stats.brightness)
    


    def report_df_contents(title,df):
        print(title)
        print("=== RICH DF CHECK ===")
        print("Number of rows:", len(df))
        print("Columns:", df.columns.tolist())
        j = random.randint(0, 4)
        if len(df) > j:
            row = df.iloc[j]
            print("Row", j,"Image_ID:", row.get('Image_ID'))
            print("Row", j,"File_Name:", row.get('File_Name'))
            print("Row", j,"Geometry:", row.get('Geometry'))
            print("Row", j,"Original_Image type:", type(row.get('Original_Image')))
            print("Row", j,"Image_Stats:", row.get('Image_Stats'))







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
        # print(images_pd.head(rows).to_string(index=False))

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



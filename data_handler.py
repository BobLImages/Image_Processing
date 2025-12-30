# data_handler.py
import os
import sqlite3
import pandas as pd
import time
import numpy as np
from typing import Optional
from pathlib import Path
from file_record import EventPathGroup







 
#     def load_image(self, file_name):
#         print(file_name)
#         image = cv2.imread(file_name)
#         return image

#     def resize_image_for_display(self, O_image):
#         scale_factor  = self. get_scale(O_image)
        
#         aspect_ratio = O_image.shape[0]/O_image.shape[1]
#         # print(f'Aspect Ratio: {aspect_ratio}')

#         RO_image_height = int(O_image.shape[0] * scale_factor)
#         RO_image_width = int(O_image.shape[1] * scale_factor)          

#         RO_image_dim = (RO_image_width, RO_image_height)
#         RO_image= cv2.resize(O_image,RO_image_dim, interpolation = cv2.INTER_AREA).astype(np.uint8)

#         return (RO_image,scale_factor)

#     def get_scale(self, O_image):
        
#         if O_image.shape[1] < O_image.shape[0]:
#             scale_factor = 896/ O_image.shape[1]
#         elif O_image.shape[1] > O_image.shape[0]:
#             scale_factor = 2016/ O_image.shape[1]
#         else:
#             scale_factor = 1

#         return scale_factor


#     def load_images(self):


#         images_for_loading = []
#         counter = 0
#         enhanced_actions = False
#         batch_size = 25
#         num_batches = len(self.file_objects) // batch_size + 1
#         total_rec_ctr = 0

#         print(f'Batches: {num_batches}')


#         for batch_num in range(num_batches):
#             batch_files = self.file_objects[batch_num * batch_size: (batch_num + 1) * batch_size]

#             for counter, file_object in enumerate(batch_files):
#                 fname = file_object.full_path  # Access the full path if 'file_object' has this attribute
#             # for counter, fname in enumerate(batch_files):
#             #     print(f'Batch Files: {batch_files}')
#                 O_image = self.load_image(fname)
#                 RO_image, scale_factor = self.resize_image_for_display(O_image)
#                 ctr = counter + batch_num * batch_size
#                 self.display_images.append(ColorImage(RO_image, ctr +1 , self.file_objects[ctr].full_path, scale_factor, O_image.shape[0], O_image.shape[1]))
#                 total_rec_ctr += 1
#                 print(f'Loaded {total_rec_ctr} of {len(self.file_objects)} Images')




















@staticmethod
def db_report(conn: Optional[sqlite3.Connection] = None, 
              sql_path: Optional[Path | str] = None) -> None:
    cursor = conn.cursor()
    print(f"\n=== DATABASE SCHEMA FOR {sql_path} ===\n")
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;")

    for table_name, create_sql in cursor.fetchall():
        print(f"Table: {table_name}")
        print(create_sql)
        print("\n-- Columns & types --")
        cursor.execute(f"PRAGMA table_info({table_name});")
        for row in cursor.fetchall():
            print(row)
        print("\n" + "-"*50)


@staticmethod
def batch_up(valid_paths,count) -> dict:
    BATCH_SIZE = 100
    
    batches = [valid_paths[i:i + BATCH_SIZE] for i in range(0, count, BATCH_SIZE)]
    return batches



# @staticmethod
# def rpt_db_audit(db_file_path: Path) -> dict:
#     db_ready = db_file_path.exists()
#     if db_ready:
#         return True
#     else:
#         return False



@staticmethod
def get_valid_image_paths(event_path: Path) -> list:   # ← Only Path
    """ The ONE TRUE JUDGE of any event directory."""
    folder = event_path
    event_name = folder.name

    # Sacred filter — only originals
    valid_paths = [
        p for p in event_path.glob("*.[jJ][pP][gG]")
        if p.is_file()
           and not p.name.lower().startswith('r_')
           and '$' not in p.name and "exposure" not in p.name
    ]

    count = len(valid_paths)
    if count  < 5:
        return False
    else: 
        return valid_paths


@staticmethod
def rpt_G7_Classified(df: pd.DataFrame, n: int = 10) -> None:

    print(f'Gentleman Farmer End-of-Day Harvest Report')
    '''Call it from anywhere — no class required.
    Adds 'review_flag' column to df when multi-offenders are found.
    '''
    if df.empty:
        print("\nNo images in catalog yet.\n")
        return

    required = ['File_Name', 'Brightness', 'Contrast', 'Contour_Info']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns for report: {missing}")
        return


    print("\n" + "═" * 70)

    print("        GENTLEMAN FARMER – END-OF-DAY HARVEST REPORT")
    print("═" * 70)

    # Top 10 lists
    print(f"\nTop {n} DARKEST (possible tombs):")
    print(df.nsmallest(n, 'Brightness')[['File_Name', 'Brightness']]
          .to_string(index=False, formatters={'Brightness': '{:.1f}'.format}))

    print(f"\nTop {n} BRIGHTEST (blown-sky suspects):")
    print(df.nlargest(n, 'Brightness')[['File_Name', 'Brightness']]
          .to_string(index=False, formatters={'Brightness': '{:.1f}'.format}))

    print(f"\nTop {n} FLATTEST (low contrast):")
    print(df.nsmallest(n, 'Contrast')[['File_Name', 'Contrast']]
          .to_string(index=False, formatters={'Contrast': '{:.1f}'.format}))

    print(f"\nTop {n} BLURRIEST (low center-crop edges):")
    print(df.nsmallest(n, 'Contour_Info')[['File_Name', 'Contour_Info']]
          .to_string(index=False))

    print(f"\nTop {n} SHARPEST (your heroes):")
    print(df.nlargest(n, 'Contour_Info')[['File_Name', 'Contour_Info']]
          .to_string(index=False))

    # Multi-offenders
    print(f"\n{'─' * 70}")
    print("MULTI-CATEGORY OFFENDERS – FLAGGED FOR CURSORY REVIEW")
    print("These hit 2+ red flags. Almost certainly garbage… but we still glance.")
    print(f"{'─' * 70}")

    TOO_DARK   = 35
    TOO_BRIGHT = 235
    TOO_FLAT   = 25
    TOO_BLURRY = 80

    flags = df.copy()
    flags["dark"]   = flags["Brightness"] < TOO_DARK
    flags["bright"] = flags["Brightness"] > TOO_BRIGHT
    flags["flat"]   = flags["Contrast"]   < TOO_FLAT
    flags["blurry"] = flags["Contour_Info"]   < TOO_BLURRY
    flags["flag_count"] = flags[["dark", "bright", "flat", "blurry"]].sum(axis=1)

    multi = flags[flags["flag_count"] >= 2].sort_values(
        ["flag_count", "File_Name"], ascending=[False, True]
    )

    if multi.empty:
        print("  None found today. A clean harvest.")
    else:
        def sins(row):
            s = []
            if row.dark:   s.append("TOMB")
            if row.bright: s.append("BLOWN")
            if row.flat:   s.append("FLAT")
            if row.blurry: s.append("BLUR")
            return ", ".join(s)

        multi = multi.copy()
        multi["reason"] = multi.apply(sins, axis=1)

        # Permanently flag in the original df
        if "review_flag" not in df.columns:
            df["review_flag"] = False
        df.loc[df.index.isin(multi.index), "review_flag"] = True

        for _, row in multi.iterrows():
            print(f"  {row['File_Name']:<20}  {row['flag_count']} flags → {row['reason']}")

        print(f"\n  → {len(multi)} frames now have review_flag = True")
        print(f"    Jump to them with 'f' key anytime.")

    print("═" * 70 + "\n")    



class DataHandler:

    CLASS_DIR_NAMES = {  
        'A': "Aesthetics",
        'B': "Reject",
        'D': "Duplicate",
        'G': "Accept"
    }


    file_stats = {
        "Valid Files": 0,
        "Images": 0,
        "Good Directory Files Count": 0,
        "Bad Directory Files Count": 0,
        "Aesthetics Directory Files Count": 0,
        "Duplicate Directory Files Count": 0,
        
        "Unclassified Dataframe Images Count": 0,
        "Good Dataframe Images": 0,
        "Bad Dataframe Images": 0,
        "Aesthetics Dataframe Images Count": 0,
        "Duplicate Dataframe Images Count": 0,
        
        "Missing Good Directory": 0,
        "Invalid Good Directory":0,
        "Missing Bad Directory": 0,
        "Invalid Bad Directory":0,
        "Missing Aesthetics Directory": 0,
        "Invalid Aesthetics Directory":0,
        "Missing Duplicate Directory": 0,
        "Invalid Duplicate Directory":0

    }



    def __init__(self, image_path: Path, db_file_path: Path):
        # ... paths setup ...
        # The ONE truth — a group that holds the family
        self.paths = ImagePathGroup(
            image_path = image_path,
            db_file_path = db_file_path
        )

        

        print(f"Event: {self.paths.image_path.name}")
        print(f"Images dir: {self.paths.image_path}")
        print(f"DB file:    {self.paths.db_file_path.name}")
        print(f"DB dir:     {self.paths.db_file_path.parent}")
        print(f"Accept dir: {self.paths.accept_dir}")
        print(f"Reject dir:   {self.paths.reject_dir}")
        print(f"Duplicate dir: {self.paths.duplicate_dir}")
        print(f"Aesthetics dir:   {self.paths.aesthetics_dir}")



        # print("\nPausing 5 seconds — admire your work...")
        # time.sleep(5)
        # print("...and we're back. Handiwork confirmed.")

        self.table_sheet = "Sheet_1"
 
        self.db_ready = rpt_db_audit(self.paths.db_file_path)
        print(f'Ready: {self.db_ready} with {self.paths.db_file_path} ')


    def load_from_db(self):
        local_images_pd = None
        
        conn = sqlite3.connect(self.paths.db_file_path)
        db_report(conn,self.paths.db_file_path)
        cursor = conn.cursor() 

        local_images_pd = pd.read_sql(f"SELECT * FROM {self.table_sheet}", conn)
        rpt_G7_Classified(local_images_pd)
        conn.close()
        
        print(f"Loaded {len(local_images_pd)} records from DB.")
        return local_images_pd


    def load_from_obj(self):
        local_images_pd = None
        
        
 

















        print(f"Loaded {len(local_images_pd)} records from objs.")
        return local_images_pd













def harvest_event_images(self, valid_files: list[Path]):
        """
        The new heart. Pure. Simple. Immortal.
        Takes the list of valid files from the audit.
        Processes in batches. Going through Feeds GUI. Feeds DB.
        """
        BATCH_SIZE = 50
        # self.display_images.clear()  # fresh start

        print(f"\n{'='*70}")
        print(f"  HARVESTING: {self.event_name}")
        print(f"  → {total_count:,} images in {len(range(0, total_count, BATCH_SIZE))} batches")
        print(f"{'='*70}")

        for i in range(0, total_count, BATCH_SIZE):
            batch_paths = valid_files[i:i + BATCH_SIZE]
            batch_colorimages = []

            for idx, path in enumerate(batch_paths):
                global_idx = i + idx + 1  # 1-based for display

                try:
                    # Load once
                    full_img = cv2.imread(str(path))
                    if full_img is None:
                        print(f"  Failed to load: {path.name}")
                        continue

                    # Resize for display
                    display_img, scale = self.resize_for_display(full_img)

                    # Create the ONE TRUE ColorImage
                    ci = ColorImage(
                        image=display_img,
                        index=global_idx,
                        full_path=str(path),
                        scale_factor=scale,
                        original_height=full_img.shape[0],
                        original_width=full_img.shape[1]
                    )

                    # Feed GUI
                    self.display_images.append(ci)

                    # Collect for DB (optional — only if saving)
                    batch_colorimages.append(ci)

                    print(f"  Loaded {global_idx}/{total_count}: {path.name}")

                except Exception as e:
                    print(f"  ERROR on {path.name}: {e}")

            # Optional: save batch to DB here if you want streaming immortality
            # if self.saving_to_db:
            #     df = self.to_dataframe(batch_colorimages)
            #     DBManager.save_batch(self.db_path, df)

        print(f"{'='*70}")
        print(f"  HARVEST COMPLETE: {len(self.display_images)} images ready")
        print(f"{'='*70}\n")


    # def get_valid_original_files(cls):

    #     valid_files = []
    #     # Step 1: List all files in the directory
    #     parent_directory_files = set(os.listdir(cls.dir_path))
    #     print(f'Parent Directory File from function: {parent_directory_files}')
    #     for file in parent_directory_files:
    #         if file.endswith('.JPG') and '$' not in file and 'r_' not in file:
    #             valid_files.append(file)
    #     cls.file_stats["Valid Files"] = len(valid_files)
    #     print(f'Number of Valid Files: {len(valid_files)}')        
    #     # verify_dataframe_images_with_originals(valid_files)        

    #     return valid_files


    # def get_file_stats(self):
    #     good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = self.data_handler.get_subdir_files()

    # def get_classification_path(self, classification: str) -> str:
    #     return self.class_dirs.get(classification, self.images_dir)
                                  
    # def get_folder_name(self):
    #     return os.path.basename(self.folder_path)  # ← os stays HERE

 
    
    # def save_to_db(self, df):
    #     conn = sqlite3.connect(self.db_path)
    #     df.to_sql("images", conn, if_exists="replace", index=False)
    #     conn.close()
    #     print(f"Saved {len(df)} records to DB.")

    # def verify_sync(self, valid_files,local_images_pd):
    #     if local_images_pd is None or local_images_pd.empty:
    #         print("No DB data to verify.")
    #         return 0

    #     db_files = set(local_images_pd['File_Name'].apply(
    #         lambda x: os.path.join(self.folder_path, os.path.basename(x))
    #     ))
    #     disk_files = set(valid_files)
    #     missing_in_db = disk_files - db_files
    #     missing_on_disk = db_files - disk_files

    #     print(f"Missing in DB: {len(missing_in_db)}")
    #     print(f"Missing on Disk: {len(missing_on_disk)}")

    #     return len(disk_files & db_files)  # Matches



    # def run_full_load_pipeline(self, folder):
    #     results = {}
    #     start_total = time.time()

    #     # === 1. Load DB ===
    #     start = time.time()
    #     df = self.load_from_db()
    #     results['load'] = time.time() - start

    #     if df.empty:
    #         return "No data", results

    #     # === 2. Sync ===
    #     start = time.time()
    #     valid_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    #     matches = self.verify_sync(valid_files, df)  # ← Pass df
    #     results['sync'] = time.time() - start

    #     # === 3. Classification ===
    #     start = time.time()
    #     class_summary = self.get_classification_summary(df)
    #     results['class'] = time.time() - start

    #     report = (
    #         f"Load: {results['load']:.3f}s | "
    #         f"Sync: {results['sync']:.3f}s | "
    #         f"Class: {results['class']:.3f}s | "
    #         # f"Display: {results['display']:.3f}s | "
    #         # f"Total: {total:.3f}s | "
    #         f"Files: {matches}/{len(valid_files)} | "
    #         f"{class_summary}"
    #     )
    #     return report, results


    # def get_classification_summary(self, df: pd.DataFrame) -> str:
    #     if df.empty:
    #         return "No data"
    #     counts = df['Classification'].value_counts().to_dict()
    #     return " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


    # ''' Old methods to determine file stats '''
    


    # @classmethod
    # def reset_classified_files(cls,filing_df):

    #     # Clear the contents of classified directories
    #     for directory in [cls.accepted_dir, cls.rejected_dir, cls.aesthetics_dir, cls.duplicate_dir]:
    #         for filename in os.listdir(directory):
    #             file_path = os.path.join(directory, filename)
    #         # List comprehension to create a list of file paths in the specified directory
    #         #file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    #             try:
    #                 if os.path.isfile(file_path):
    #                     os.remove(file_path)
    #                 elif os.path.isdir(file_path):
    #                     shutil.rmtree(file_path)
    #             except Exception as e:
    #                 print(f"Error: {e}")
 

    #     # Repopulate classified directories based on images_pd DataFrame
        
    #     #Fill empty classification directories

    #     print("Fill empty classification directories")
    #     for index, row in filing_df.iterrows():
    #         classification = row['Classification']
    #         file_name = os.path.basename(row['File_Name'])
    #         destination_dir = None

    #         if classification == 'G':
    #             destination_dir = cls.accepted_dir
    #         elif classification == 'B':
    #             destination_dir = cls.rejected_dir
    #         elif classification == 'A':
    #             destination_dir = cls.aesthetics_dir
    #         elif classification == 'D':
    #             destination_dir = cls.duplicate_dir
    #         else:
    #             #print(f"Invalid classification for file: {file_name}")
    #             continue

    #         source_file_path = os.path.join(cls.parent_dir, file_name)
    #         destination_file_path = os.path.join(destination_dir, f"{classification}{file_name}")

    #         try:
    #             shutil.copy2(source_file_path, destination_file_path)
    #         except Exception as e:
    #             print(f"Error copying file {file_name}: {e}")

    #     print(f"Classified files have been reset.")


    # @classmethod
    # def recalculate_file_stats(cls, images_pd):
        
    #     # print(f'Here***************************************************************************')
    #     # Files in subdirectories
    #     good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = cls.get_subdir_files()       
    #     print(good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files)

    #     # Rows by classification in dataframe
    #     good_df_rows, bad_df_rows, aesthetics_df_rows, duplicate_df_rows, unclassified_df_rows = cls.get_rows_by_classification(images_pd)        

    #     directories = {
    #         "Good Directory": (good_df_rows, good_directory_files),
    #         "Bad Directory": (bad_df_rows, bad_directory_files),
    #         "Aesthetics Directory": (aesthetics_df_rows, aesthetics_directory_files),
    #         "Duplicate Directory": (duplicate_df_rows, duplicate_directory_files)
    #     }

    #     for directory, (df_rows, dir_files) in directories.items():
    #         print(directory,(len(df_rows),len(dir_files)))
    #         missing_files = len(df_rows) - len(dir_files)
    #         invalid_files = len(dir_files) - len(df_rows)
    #         cls.file_stats[f"Missing from {directory}"] = missing_files
    #         cls.file_stats[f"Invalid in {directory}"] = invalid_files
    #         if missing_files > 0 or invalid_files > 0:
    #             filing_df = images_pd[['File_Name', 'Classification']]
    #             return filing_df

    #         filing_df = pd.DataFrame()
    #         return filing_df


    # @classmethod
    # def get_rows_by_classification(cls, images_pd):
    #     class_counts = images_pd['Classification'].value_counts()
        
    #     cls.file_stats["Unclassified Dataframe Images Count"] = class_counts.get('U', 0)
    #     cls.file_stats["Good Dataframe Images Count"] = class_counts.get('G', 0)
    #     cls.file_stats["Bad Dataframe Images Count"] = class_counts.get('B', 0)
    #     cls.file_stats["Aesthetics Dataframe Images Count"] = class_counts.get('A', 0)
    #     cls.file_stats["Duplicate Dataframe Images Count"] = class_counts.get('D', 0)

    #     # Generate lists of file names for each row of the dataframe corrected for classification 
    #     good_df_rows = set(images_pd.loc[images_pd['Classification'] == 'G', 'File_Name'].apply(lambda x: 'G' + os.path.basename(x)))
    #     bad_df_rows = set(images_pd.loc[images_pd['Classification'] == 'B', 'File_Name'].apply(lambda x: 'B' + os.path.basename(x)))
    #     aesthetics_df_rows = set(images_pd.loc[images_pd['Classification'] == 'A', 'File_Name'].apply(lambda x: 'A' + os.path.basename(x)))
    #     duplicate_df_rows = set(images_pd.loc[images_pd['Classification'] == 'D', 'File_Name'].apply(lambda x: 'D' + os.path.basename(x)))
    #     unclassified_df_rows = set(images_pd.loc[images_pd['Classification'] == 'U', 'File_Name'].apply(lambda x: '' + os.path.basename(x)))

    #     return good_df_rows, bad_df_rows, aesthetics_df_rows,duplicate_df_rows, unclassified_df_rows




    # @classmethod
    # def get_valid_original_files(cls):

    #     valid_files = []
    #     # Step 1: List all files in the directory
    #     parent_directory_files = set(os.listdir(cls.dir_path))
    #     print(f'Parent Directory File from function: {parent_directory_files}')
    #     for file in parent_directory_files:
    #         if file.endswith('.JPG') and '$' not in file and 'r_' not in file:
    #             valid_files.append(file)
    #     cls.file_stats["Valid Files"] = len(valid_files)
    #     print(f'Number of Valid Files: {len(valid_files)}')        
    #     # verify_dataframe_images_with_originals(valid_files)        

    #     return valid_files


    #     # Repopulate classified directories based on images_pd DataFrame
    #     for index, row in filing_df.iterrows():
    #         classification = row['Classification']
    #         file_name = os.path.basename(row['File_Name'])
    #         destination_dir = None

    #         if classification == 'G':
    #             destination_dir = cls.accepted_dir
    #         elif classification == 'B':
    #             destination_dir = cls.rejected_dir
    #         elif classification == 'A':
    #             destination_dir = cls.aesthetics_dir
    #         elif classification == 'D':
    #             destination_dir = cls.duplicate_dir
    #         else:
    #             print(f"Invalid classification for file: {file_name}")
    #             continue

    #         source_file_path = os.path.join(cls.parent_dir, file_name)
    #         destination_file_path = os.path.join(destination_dir, f"{classification}{file_name}")

    #         try:
    #             shutil.copy2(source_file_path, destination_file_path)
    #         except Exception as e:
    #             print(f"Error copying file {file_name}: {e}")

    #     print(f"Classified files have been reset.")



    # @classmethod
    # def recalculate_file_stats(cls, df: pd.DataFrame, folder_path: str) -> Dict[str, int]:
    #     stats = {}
        
    #     # DB counts
    #     df_counts = df['Classification'].value_counts()
    #     for code, name in [('G', 'Good'), ('B', 'Bad'), ('A', 'Aesthetics'), ('D', 'Duplicates')]:
    #         stats[f"{name} in DB"] = df_counts.get(code, 0)

            
    #     df_counts = df['Classification'].value_counts()
    #     for code, name in self.CLASS_DIRS.items():
    #         stats[f"{name} in DB"] = df_counts.get(code, 0)


    #    # Disk counts
    #     for code, name in [('G', 'Good'), ('B', 'Bad'), ('A', 'Aesthetics'), ('D', 'Duplicates')]:
    #         dir_path = os.path.join(folder_path, name)
    #         if os.path.exists(dir_path):
    #             files = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
    #             stats[f"{name} on Disk"] = len(files)
    #             stats[f"Missing in {name}"] = max(0, stats[f"{name} in DB"] - len(files))
    #             stats[f"Extra in {name}"] = max(0, len(files) - stats[f"{name} in DB"])

    #     return stats




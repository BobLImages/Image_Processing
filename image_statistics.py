# statistics.py

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class FileStats:
    """Sync and catalog status — not pixel stats."""
    files_on_disk: int = 0                    # total valid images in event_path
    rows_in_db: int = 0                       # total rows in catalog DB

    accepted_files: int = 0                     # count of classification 'A'
    rejected_files: int = 0                     # 'R'
    duplicate_files: int = 0                     # 'D'
    unknown_soldier_files: int = 0                 # anything else
 
    accepted_rows: int = 0                     # count of classification 'A'
    rejected_rows: int = 0                     # 'R'
    duplicate_rows: int = 0                     # 'D'
    unknown_soldier_rows: int = 0    


    new_on_disk: int = 0                      # files on disk not in DB
    missing_from_disk: int = 0                # files in DB not on disk

    # Optional: derived
    @property
    def in_sync(self) -> bool:
        return self.new_on_disk == 0 and self.missing_from_disk == 0

    @property
    def total_classified(self) -> int:
        return self.classified_A + self.classified_R + self.classified_D + self.classified_other

@dataclass
class  ImageStatistics:
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    laplacian: Optional[float] = None
    harris_corners: Optional[int] = None
    haze_factor: Optional[float] = None
    variance: Optional[float] = None
    shv: Optional[float] = None














class CameraSettings:
    def __init__(self, full_path):

        self.exposure = 0
        self.fstop = 0
        self.iso = 0
        self.focal_length = 0
        self.bodyserialnumber = []
        self.datetimeoriginal = []
        self.get_exif_data(full_path)



    def get_exif_data(self, full_path):
        im = open(full_path,'rb')
        tags = exifread.process_file(im)
        for tag in tags.keys():
            if tag in "EXIF ExposureTime":
                result = str(tags[tag])
                if '/' in result:
                    new_result, x = result.split('/')
                    self.exposure = int(new_result)/int(x)
                else:
                    self.exposure = int(result)    

            if tag in "EXIF FNumber":
                result = str(tags[tag])
                if '/' in result:
                    new_result, x = result.split('/')
                    self.fstop = int(new_result)/int(x)
                else:
                    self.fstop = int(result)    

            if tag in "EXIF ISOSpeedRatings":
                result = str(tags[tag])
                self.iso = int(result)

            if tag in 'EXIF DateTimeOriginal':
                result = str(tags[tag])
                self.datetimeoriginal = result
                #print(self.datetimeoriginal)

            if tag in "EXIF BodySerialNumber":
                result = str(tags[tag])
                self.bodyserialnumber = result
                #print(self.bodyserialnumber)

            if tag in "EXIF FocalLength":
                result = str(tags[tag])
                self.focal_length = result
                #print(self.focal_length)        
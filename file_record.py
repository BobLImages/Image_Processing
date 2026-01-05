# # file_record.py


#                 EXAMPLES USED IN FOLOWING CLASSES
# event_dir = K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football
# db_root   = D:\Image Data Files sql
# image     = K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\5Q2A0690.JPG



from dataclasses import dataclass
from pathlib import Path
from image_functions import ImageFunctions as IF
from color_image import ColorImage 
from rpt_dbg_tst import RTD




@staticmethod
def get_disk_files(event_path: Path) -> list:
    valid = find_candidate_images(event_path)
    if len(valid) > 4:
        return valid
    return None

@staticmethod
def find_candidate_images(event_path: Path) -> list[Path]:
        # Sacred filter — only originals
    return [
        p for p in event_path.glob("*.[jJ][pP][gG]")
        if p.is_file()
           and not p.name.lower().startswith('r_')
           and '$' not in p.name and "exposure" not in p.name
    ]


@dataclass(frozen=True)
class ImagePathGroup:
    image_path: Path   # K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\5Q2A0690.JPG
    db_file_path: Path # D:\Image Data Files sql\2025-12-06 Jefferson Forest vs Varina Football.db

    @property
    def name(self) -> str:
        return self.image_path.name
        # → "5Q2A0690.JPG"

    @property
    def stem(self) -> str:
        return self.image_path.stem
        # → "5Q2A0690"

    @property
    def suffix(self) -> str:
        return self.image_path.suffix.lower()
        # → ".jpg"

    @property
    def parent(self) -> Path:
        return self.image_path.parent
        # → K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football

    @property
    def event_name(self) -> str:
        return self.parent.name
        # → "2025-12-06 Jefferson Forest vs Varina Football"

    @property
    def size_bytes(self) -> int:
        return self.image_path.stat().st_size
        # → e.g. 4821934

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)
        # → e.g. 4.6

    @property
    def modified_time(self) -> float:
        return self.image_path.stat().st_mtime
        # → e.g. 1702439912.0

    # — The voice of truth —
    def __repr__(self) -> str:
        return f"FileRecord({self.index}: {self.name} — {self.size_mb:.1f}MB)"

    def __str__(self) -> str:
        return self.__repr__()



@dataclass(frozen=True)
class EventPathGroup:
    event_path: Path   # K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football
    db_root: Path     # D:\Image Data Files sql

    @property
    def event_name(self) -> str:
        return self.event_path.name
        # → "2025-12-06 Jefferson Forest vs Varina Football"

    @property
    def catalog_db(self) -> Path:
        return self.db_root / f"{self.event_path.name}.db"
        # → D:\Image Data Files sql\2025-12-06 Jefferson Forest vs Varina Football.db

    @property
    def accept_dir(self) -> Path:
        return self.event_path / "Accept"
        # → K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\Accept

    @property
    def reject_dir(self) -> Path:
        return self.event_path / "Reject"
        # → K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\Reject

    @property
    def duplicate_dir(self) -> Path:
        return self.event_path / "Duplicate"
        # → K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\Duplicate

    @property
    def aesthetics_dir(self) -> Path:
        return self.event_path / "Aesthetics"
        # → K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\Aesthetics



@dataclass(frozen=True)
class FrameworkPathGroup:
    db_root: Path  # D:\Image Data Files sql

    @property
    def notes_root(self) -> Path:
        return self.db_root / "Notes"
        # → D:\Image Data Files sql\Notes

    @property
    def notes_db(self) -> Path:
        return self.db_root / "Notes" / "notes.db"
        # → D:\Image Data Files sql\Notes\notes.db

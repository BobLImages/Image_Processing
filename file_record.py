# # file_record.py


'''
                EXAMPLES USED IN FOLOWING CLASSES
event_dir = K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football
db_root   = D:\Image Data Files sql
image     = K:\Year 2025\2025-12-06 Jefferson Forest vs Varina Football\5Q2A0690.JPG
'''
from dataclasses import dataclass
from pathlib import Path

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



# @dataclass
# class FrameworkPaths:
#     db_root: Path

#     # Derived paths as properties (very common in dataclasses)
#     @property
#     def notes_root(self) -> Path:
#         return self.db_root / "notes"

#     @property
#     def notes_db(self) -> Path:
#         return self.note_root / "notes.db"

#     # @property
#     # def thumbnails(self) -> Path:
#     #     return self.db_root / "thumbnails"

#     # Instance method — small behavior directly tied to the data
#     def ensure_exists(self) -> None:
#         """Create required directories if they don't exist."""
#         self.db_root.mkdir(parents=True, exist_ok=True)
#         self.note_root.mkdir(parents=True, exist_ok=True)
#         # self.thumbnails.mkdir(parents=True, exist_ok=True)

#     # Or make it automatic after init
#     def __post_init__(self) -> None:
#         self.ensure_exists()


#     # Static method — useful utility not tied to an instance
#     @staticmethod
#     def default_location() -> Path:
#         """Return the default framework location (e.g., in user's home)."""
#         return Path.home() / ".my_image_app" / "framework"

#     # Class method — alternative constructor
#     @classmethod
#     def from_default(cls) -> "FrameworkPaths":
#         return cls(cls.default_location())



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

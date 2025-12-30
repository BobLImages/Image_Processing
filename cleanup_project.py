#cleanup_project.py
# Run this once — it cleans your project and then archives itself!
# organize_archive.py
# Run this once in your BobLImages root — it cleans and organizes everything

from pathlib import Path
import shutil

# Project root = folder this script lives in
ROOT = Path(__file__).parent.resolve()
ARCHIVE = ROOT / "archive"
ARCHIVE.mkdir(exist_ok=True)

# === CORE FILES TO KEEP IN ROOT ===
KEEP = {
    "aux_wind.py",
    "color_image.py",
    "data_handler.py",
    "file_record.py",
    "harvester.py",
    "image_catalog.py",
    "image_functions.py",
    "ipa_controller.py",
    "ipa_gui.py",
    "main.py",
    "mask_defs.py",
    "mask_factory.py",
    "mask_repository.py",
    "notes_repository.py",
    "rpt_dbg_tst.py",
}

# === ARCHIVE SUBFOLDERS AND FILES ===
SUBFOLDERS = {
    "titan_project": [
        "TITAN_SPAWN.py", "titan_image_processor.py", "Titan.py", "titan_data_handler.py"
    ],
    "physics_playground": [
        "ball_objects.py", "base.py", "guinea_pig.py"
    ],
    "geometry_experiments": [
        "700x700.py", "dominant.py", "present_thumbnails.py"
    ],
    "old_image_processing": [
        "image_processing_app.py", "image_processor.py", "segment_class.py", "mask_class.py"
    ],
    "misc_experiments": [
        "Snowglobe.py", "mandelb.py", "scrape.py", "crapture_video.py",
        "random_forest.py", "rf_class.py", "big color.py", "boba.py",
        "colorimage.py", "color_image_class.py", "image_catalog (1).py",
        "ipa_app.py", "ipa_ui.py", "image_functions -.py", "image_functions-.py",
        "image_functi_2.py", "image_path_group_class.py", "imp.py",
        "make_refactor_log.py", "navigation.py", "notes.py", "showimage.py",
        # add any stragglers here
    ],
}

# Create subfolders
for folder in SUBFOLDERS:
    (ARCHIVE / folder).mkdir(exist_ok=True)

# Move files
moved = 0
for item in ROOT.iterdir():
    if item.name == "__pycache__":
        shutil.rmtree(item, ignore_errors=True)
        print(f"Deleted: {item.name}")
        continue

    if item.name == ARCHIVE.name:
        continue

    if item.name == Path(__file__).name:
        continue  # don't move self yet

    if item.is_file() and item.suffix == ".py":
        if item.name in KEEP:
            continue

        placed = False
        for folder, files in SUBFOLDERS.items():
            if item.name in files:
                dest = ARCHIVE / folder / item.name
                shutil.move(str(item), dest)
                print(f"Moved: {item.name} → archive/{folder}/")
                moved += 1
                placed = True
                break

        if not placed:
            dest = ARCHIVE / "misc_experiments" / item.name
            shutil.move(str(item), dest)
            print(f"Moved: {item.name} → archive/misc_experiments/")
            moved += 1

print(f"\nCleanup complete — {moved} files archived")

# Self-archive
self_dest = ARCHIVE / "organize_archive.py"
shutil.move(__file__, self_dest)
print(f"Self-archived → archive/organize_archive.py")
print("Root is now clean. You're welcome.")
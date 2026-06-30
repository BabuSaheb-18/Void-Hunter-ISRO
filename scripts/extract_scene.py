"""
==============================================================================
VHNet v2 (Multi-Task)
Automatic Landsat Scene Extraction

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Extracts downloaded Landsat 8/9 Level-2 archives into Extracted_Scenes.
Verifies the presence of required spectral bands (RGB, NIR, Thermal).
==============================================================================
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import shutil
import tarfile
from tqdm import tqdm

from config import (
    RAW_SCENES,
    EXTRACTED_SCENES,
)

# Required bands for VHNet v2 Multi-Task Pipeline
REQUIRED_BANDS = ["_SR_B2", "_SR_B3", "_SR_B4", "_SR_B5", "_ST_B10"]

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

EXTRACTED_SCENES.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND ARCHIVES
# ============================================================

TAR_FILES = sorted(
    list(RAW_SCENES.glob("*.tar")) +
    list(RAW_SCENES.glob("*.tar.gz"))
)

# ============================================================
# EXTRACTION ENGINE
# ============================================================

def extract_archives():
    print("\n" + "=" * 70)
    print("VHNet v2 - LANDSAT SCENE EXTRACTION")
    print("=" * 70)
    print(f"Raw Folder      : {RAW_SCENES}")
    print(f"Output Folder   : {EXTRACTED_SCENES}")
    print(f"Archives Found  : {len(TAR_FILES)}")
    print("=" * 70)

    for archive in tqdm(TAR_FILES, desc="Extracting"):
        if archive.name.endswith(".tar.gz"):
            scene_name = archive.name[:-7]
            mode = "r:gz"
        else:
            scene_name = archive.stem
            mode = "r"

        output_folder = EXTRACTED_SCENES / scene_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Check if already extracted
        tif_files = list(output_folder.glob("*.TIF")) + list(output_folder.glob("*.tif"))
        if len(tif_files) > 0:
            continue

        try:
            with tarfile.open(archive, mode) as tar:
                tar.extractall(output_folder)
        except Exception as e:
            print(f"\n[!] Extraction Failed : {archive.name}")
            print(e)

# ============================================================
# CLEAN NESTED FOLDERS
# ============================================================

def clean_extracted_folders():
    print("\nCleaning extracted folders and flattening directory structure...")
    
    for scene_folder in EXTRACTED_SCENES.iterdir():
        if not scene_folder.is_dir():
            continue

        items = list(scene_folder.iterdir())
        if len(items) != 1:
            continue

        nested = items[0]
        if not nested.is_dir():
            continue

        for file in nested.iterdir():
            shutil.move(str(file), str(scene_folder))
        shutil.rmtree(nested)

# ============================================================
# VERIFY EXTRACTION & BANDS
# ============================================================

def verify_extraction():
    valid_scenes = 0
    total_tif = 0
    missing_band_warnings = 0

    print("\nVerifying required Landsat bands...")

    for scene_folder in EXTRACTED_SCENES.iterdir():
        if not scene_folder.is_dir():
            continue

        tif_files = list(scene_folder.glob("*.TIF")) + list(scene_folder.glob("*.tif"))
        
        if len(tif_files) > 0:
            valid_scenes += 1
            total_tif += len(tif_files)
            
            # Check for VHNet v2 required bands
            filenames = [f.name for f in tif_files]
            missing_bands = [b for b in REQUIRED_BANDS if not any(b in fname for fname in filenames)]
            
            if missing_bands:
                print(f"  [!] Warning: {scene_folder.name} is missing bands: {missing_bands}")
                missing_band_warnings += 1

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Extracted Scenes : {valid_scenes}")
    print(f"Total TIF Files  : {total_tif}")
    print(f"Band Warnings    : {missing_band_warnings} (Scenes missing required RGB/NIR/Thermal data)")
    print(f"Output Folder    : {EXTRACTED_SCENES}")
    print("=" * 70)

# ============================================================
# MAIN
# ============================================================

def main():
    if len(TAR_FILES) == 0:
        print("\n[!] No .tar or .tar.gz archives found in Raw_Scenes.")
        return
        
    extract_archives()
    clean_extracted_folders()
    verify_extraction()

if __name__ == "__main__":
    main()
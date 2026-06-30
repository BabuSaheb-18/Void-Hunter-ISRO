"""
==============================================================================
VHNet v2 (Multi-Task)
Automatic Patch & Mask Generator

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Creates paired RGB, IR, and Semantic Mask patches from Landsat 8/9 scenes.
Implements automated Pseudo-Labeling using Spectral Indices (NDVI / NDWI).
==============================================================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

from config import (
    EXTRACTED_SCENES,
    TRAIN_RGB_DIR, TRAIN_IR_DIR, TRAIN_MASK_DIR,
    VAL_RGB_DIR, VAL_IR_DIR, VAL_MASK_DIR,
    TEST_RGB_DIR, TEST_IR_DIR, TEST_MASK_DIR,
    PATCH_SIZE,
    PATCH_STRIDE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RGB_BANDS,
    IR_BAND,
)

# Standard Landsat 8/9 Near-Infrared Band (Used for Semantic Masks)
NIR_BAND = "_SR_B5" 

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(image):
    image = image.astype(np.float32)
    image = np.nan_to_num(image)
    p2 = np.percentile(image, 2)
    p98 = np.percentile(image, 98)
    image = np.clip(image, p2, p98)
    image = (image - p2) / (p98 - p2 + 1e-8)
    image = (image * 255).astype(np.uint8)
    return image

# ============================================================
# READ & FIND BANDS
# ============================================================

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1)

def find_band(scene_path, band):
    for file in scene_path.glob("*.TIF"):
        if band in file.name:
            return file
    for file in scene_path.glob("*.tif"):
        if band in file.name:
            return file
    return None

# ============================================================
# CREATE RGB
# ============================================================

def create_rgb(scene_path):
    red = find_band(scene_path, RGB_BANDS["red"])
    green = find_band(scene_path, RGB_BANDS["green"])
    blue = find_band(scene_path, RGB_BANDS["blue"])

    if red is None or green is None or blue is None:
        return None

    r_img = normalize(read_band(red))
    g_img = normalize(read_band(green))
    b_img = normalize(read_band(blue))

    return np.dstack((r_img, g_img, b_img))

# ============================================================
# CREATE IR
# ============================================================

def create_ir(scene_path):
    thermal = find_band(scene_path, IR_BAND)
    if thermal is None:
        return None
    return normalize(read_band(thermal))

# ============================================================
# CREATE SEMANTIC MASK (AUTOMATED PSEUDO-LABELING)
# ============================================================

def create_mask(scene_path, rgb_img):
    """
    Generates a 4-class Semantic Mask (0: Background, 1: Water, 2: Veg, 3: Man-made)
    Uses strict Remote Sensing spectral indices if NIR is available.
    """
    height, width = rgb_img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8) # Default Class 0: Background
    
    red_file = find_band(scene_path, RGB_BANDS["red"])
    green_file = find_band(scene_path, RGB_BANDS["green"])
    nir_file = find_band(scene_path, NIR_BAND)
    
    # --- METHOD A: Scientific Spectral Indices (NDVI & NDWI) ---
    if red_file and green_file and nir_file:
        red = read_band(red_file).astype(np.float32)
        green = read_band(green_file).astype(np.float32)
        nir = read_band(nir_file).astype(np.float32)
        
        # Avoid division by zero
        ndvi_denom = (nir + red) + 1e-8
        ndwi_denom = (green + nir) + 1e-8
        
        ndvi = (nir - red) / ndvi_denom
        ndwi = (green - nir) / ndwi_denom
        
        # Assign classes based on thresholds
        mask[ndvi > 0.3] = 2  # Class 2: Vegetation
        mask[ndwi > 0.1] = 1  # Class 1: Water
        
        # Note: Class 3 (Man-made) can be complex to isolate without SWIR bands.
        # Leaving remaining as Class 0 (Background/Bare Earth).
        return mask

    # --- METHOD B: Fallback Color Thresholding ---
    print("  [!] NIR band not found. Falling back to OpenCV Color Segmentation.")
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    
    # Heuristic Green (Vegetation)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    veg_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask[veg_mask > 0] = 2
    
    # Heuristic Blue (Water)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    water_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask[water_mask > 0] = 1
    
    return mask

# ============================================================
# PATCH QUALITY
# ============================================================

def is_valid_patch(rgb_patch, ir_patch):
    if rgb_patch is None or ir_patch is None:
        return False
    if rgb_patch.shape[:2] != ir_patch.shape:
        return False
    # Drop patches with no texture/variance
    if rgb_patch.std() < 5 or ir_patch.std() < 5:
        return False
    # Drop patches that are mostly empty space / edges
    if np.mean(ir_patch == 0) > 0.60:
        return False
    return True

# ============================================================
# DATASET SPLIT
# ============================================================

rng = np.random.default_rng(42)

def get_output_folders():
    r = rng.random()
    if r < TRAIN_SPLIT:
        return TRAIN_RGB_DIR, TRAIN_IR_DIR, TRAIN_MASK_DIR
    elif r < TRAIN_SPLIT + VAL_SPLIT:
        return VAL_RGB_DIR, VAL_IR_DIR, VAL_MASK_DIR
    return TEST_RGB_DIR, TEST_IR_DIR, TEST_MASK_DIR

# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(rgb_patch, ir_patch, mask_patch, index):
    rgb_dir, ir_dir, mask_dir = get_output_folders()

    rgb_file = rgb_dir / f"rgb_{index:06d}.png"
    ir_file = ir_dir / f"ir_{index:06d}.png"
    mask_file = mask_dir / f"mask_{index:06d}.png"

    # Save RGB
    cv2.imwrite(str(rgb_file), cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2BGR))
    # Save IR (Grayscale)
    cv2.imwrite(str(ir_file), ir_patch)
    # Save Mask (Grayscale Integer Class Labels)
    cv2.imwrite(str(mask_file), mask_patch)

# ============================================================
# PATCH GENERATION
# ============================================================

def create_patches(rgb_image, ir_image, mask_image, start_index):
    height, width = ir_image.shape
    saved = 0
    skipped = 0
    index = start_index

    for y in tqdm(range(0, height - PATCH_SIZE + 1, PATCH_STRIDE), leave=False, desc="Rows"):
        for x in range(0, width - PATCH_SIZE + 1, PATCH_STRIDE):
            
            rgb_patch = rgb_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            ir_patch = ir_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            mask_patch = mask_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            if not is_valid_patch(rgb_patch, ir_patch):
                skipped += 1
                continue

            save_patch(rgb_patch, ir_patch, mask_patch, index)
            saved += 1
            index += 1

    print(f"Saved: {saved} | Skipped: {skipped}")
    return index

# ============================================================
# PROCESS SCENE
# ============================================================

def process_scene(scene_path, start_index):
    print("\n" + "=" * 70)
    print(f"Processing : {scene_path.name}")
    print("=" * 70)

    rgb = create_rgb(scene_path)
    ir = create_ir(scene_path)

    if rgb is None:
        print("RGB bands not found.")
        return start_index
    if ir is None:
        print("Thermal band not found.")
        return start_index

    if rgb.shape[:2] != ir.shape:
        raise RuntimeError("RGB and IR dimensions do not match.")

    # Generate Semantic Mask
    mask = create_mask(scene_path, rgb)

    print(f"Image Size : {rgb.shape[1]} x {rgb.shape[0]}")
    return create_patches(rgb, ir, mask, start_index)

# ============================================================
# DISCOVER SCENES
# ============================================================

def discover_scenes():
    scenes = sorted([
        scene for scene in EXTRACTED_SCENES.iterdir() if scene.is_dir()
    ])
    if len(scenes) == 0:
        raise RuntimeError("No extracted Landsat scenes found.")
    return scenes

# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset():
    scenes = discover_scenes()

    print("\n" + "=" * 70)
    print("VHNet v2 Dataset Generation (RGB + IR + Mask)")
    print("=" * 70)
    print(f"Scenes Found : {len(scenes)}")
    print("=" * 70)

    patch_index = 0
    for scene in scenes:
        patch_index = process_scene(scene, patch_index)
        
    return patch_index

# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary():
    train = len(list(TRAIN_RGB_DIR.glob("*.png")))
    val = len(list(VAL_RGB_DIR.glob("*.png")))
    test = len(list(TEST_RGB_DIR.glob("*.png")))
    total = train + val + test

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Train Patches      : {train}")
    print(f"Validation Patches : {val}")
    print(f"Test Patches       : {test}")
    print("-" * 70)
    print(f"Total Patches      : {total}")
    print("=" * 70)

# ============================================================
# MAIN
# ============================================================

def main():
    generate_dataset()
    dataset_summary()

if __name__ == "__main__":
    main()
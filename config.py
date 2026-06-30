"""
============================================================
VHNet v2 Configuration (Multi-Task)

ISRO Hackathon 2026
Team : Void Hunter

Model      : VHNet v2
Pipeline   : 256×256 IR -> 512×512 RGB + 256x256 Mask
Tasks      : Super Resolution & Semantic Segmentation
============================================================
"""
# Add this at the top of app.py
from pathlib import Path
import os
import random

import numpy as np
import torch

# ============================================================
# PROJECT
# ============================================================

PROJECT_NAME = "VHNet v2"
VERSION = "2.0"
TEAM_NAME = "Void Hunter"
COMPETITION = "ISRO Hackathon 2026 - Problem Statement 10"
DESCRIPTION = (
    "Infrared Image Colorization, 2× Super Resolution, "
    "and Object Interpretation (Semantic Segmentation)"
)
AUTHOR = "Void Hunter Team"

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if USE_CUDA else "CPU"

# ============================================================
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

# Colab
if "content" in ROOT_DIR.parts:
    DRIVE_ROOT = Path("/content/drive/MyDrive/VHNetV2")
# Windows / Local
else:
    DRIVE_ROOT = ROOT_DIR

# ============================================================
# RAW DATA
# ============================================================

RAW_SCENES = DRIVE_ROOT / "Raw_Scenes"
EXTRACTED_SCENES = DRIVE_ROOT / "Extracted_Scenes"

RAW_SCENES.mkdir(parents=True, exist_ok=True)
EXTRACTED_SCENES.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROCESSED DATASET
# ============================================================

PROCESSED_DATASET = Path("/content/local_dataset")

TRAIN_DIR = PROCESSED_DATASET / "train"
VAL_DIR = PROCESSED_DATASET / "val"
TEST_DIR = PROCESSED_DATASET / "test"

# Modality Paths
TRAIN_RGB_DIR = TRAIN_DIR / "rgb"
TRAIN_IR_DIR = TRAIN_DIR / "ir"
TRAIN_MASK_DIR = TRAIN_DIR / "mask"

VAL_RGB_DIR = VAL_DIR / "rgb"
VAL_IR_DIR = VAL_DIR / "ir"
VAL_MASK_DIR = VAL_DIR / "mask"

TEST_RGB_DIR = TEST_DIR / "rgb"
TEST_IR_DIR = TEST_DIR / "ir"
TEST_MASK_DIR = TEST_DIR / "mask"

# ============================================================
# INTERMEDIATE DATA
# ============================================================

RGB_IMAGES = DRIVE_ROOT / "RGB_Images"
IR_IMAGES = DRIVE_ROOT / "IR_Images"
TEMP_DIR = DRIVE_ROOT / "Temp"

# ============================================================
# OUTPUT
# ============================================================

CHECKPOINT_DIR = DRIVE_ROOT / "models" / "checkpoints"
RESULT_DIR = DRIVE_ROOT / "Results"
REPORT_DIR = DRIVE_ROOT / "Reports"
LOG_DIR = DRIVE_ROOT / "Logs"

PREDICTION_DIR = RESULT_DIR / "Predictions"
METRIC_DIR = RESULT_DIR / "Metrics"
FIGURE_DIR = RESULT_DIR / "Figures"

# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    EXTRACTED_SCENES,
    PROCESSED_DATASET,
    
    TRAIN_RGB_DIR,
    TRAIN_IR_DIR,
    TRAIN_MASK_DIR,

    VAL_RGB_DIR,
    VAL_IR_DIR,
    VAL_MASK_DIR,

    TEST_RGB_DIR,
    TEST_IR_DIR,
    TEST_MASK_DIR,

    RGB_IMAGES,
    IR_IMAGES,
    TEMP_DIR,

    CHECKPOINT_DIR,
    RESULT_DIR,
    REPORT_DIR,
    LOG_DIR,
    PREDICTION_DIR,
    METRIC_DIR,
    FIGURE_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_NAME = "Landsat 8/9"
SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

RGB_BANDS = {"red": "_SR_B4", "green": "_SR_B3", "blue": "_SR_B2"}
IR_BAND = "_ST_B10"

# ============================================================
# IMAGE CONFIGURATION
# ============================================================

INPUT_SIZE = 256
OUTPUT_SIZE = 512
NUM_CLASSES = 4
UPSCALE_FACTOR = 2

PATCH_SIZE = 256
PATCH_STRIDE = 128

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 3

NORMALIZE_IMAGES = True
NORMALIZATION_PERCENTILE_LOW = 2
NORMALIZATION_PERCENTILE_HIGH = 98

IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
SAVE_IMAGE_FORMAT = "png"

# ============================================================
# DATASET SPLIT
# ============================================================

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
SHUFFLE_DATASET = True

# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True
DROP_LAST = False
PERSISTENT_WORKERS = False

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "VHNet"
MODEL_VERSION = "v2"
MODEL_FULL_NAME = f"{MODEL_NAME} {MODEL_VERSION}"

INPUT_RESOLUTION = INPUT_SIZE
OUTPUT_RESOLUTION = OUTPUT_SIZE
SUPER_RESOLUTION = True

BASE_CHANNELS = 64
ENCODER_CHANNELS = (64, 128, 256, 512)
BOTTLENECK_CHANNELS = 1024
DECODER_CHANNELS = (512, 256, 128, 64)

USE_BATCH_NORM = True
USE_SKIP_CONNECTIONS = True
USE_ATTENTION = True
UPSAMPLE_MODE = "bilinear"
FINAL_ACTIVATION = "sigmoid"

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 25
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER = "AdamW"
SCHEDULER = "ReduceLROnPlateau" # Changed from CosineAnnealingLR
MIN_LEARNING_RATE = 1e-6
GRADIENT_CLIP = 1.0

# --- NEW SCHEDULER VARIABLES ADDED HERE ---
SCHEDULER_FACTOR = 0.5        # Cut learning rate by 50% when stuck
SCHEDULER_PATIENCE = 2        # Wait 2 epochs of flat validation before dropping LR
# ------------------------------------------

USE_MIXED_PRECISION = True
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15

SAVE_BEST_MODEL_ONLY = True
SAVE_EVERY_EPOCH = False
PRINT_FREQUENCY = 10

# ============================================================
# CHECKPOINTS
# ============================================================

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"
TRAINING_HISTORY_PATH = CHECKPOINT_DIR / "history.json"

# ============================================================
# LOSS
# ============================================================

# RGB Reconstruction Weights
L1_WEIGHT = 1.0
SSIM_WEIGHT = 0.50
LPIPS_WEIGHT = 0.20
EDGE_WEIGHT = 0.10

# Segmentation Multi-Task Weight
SEGMENTATION_WEIGHT = 1.0

# ============================================================
# METRICS
# ============================================================

METRICS = ["PSNR", "SSIM", "LPIPS", "FID", "MSE", "mIoU"]
PRIMARY_METRIC = "PSNR"
HIGHER_IS_BETTER = True

# ============================================================
# INFERENCE
# ============================================================

INFERENCE_BATCH_SIZE = 1
SAVE_OUTPUT_IMAGES = True
SAVE_COMPARISON = True
SAVE_METRICS = True
OUTPUT_IMAGE_FORMAT = "png"
USE_TTA = False

# ============================================================
# LOGGING
# ============================================================

LOG_FILE = LOG_DIR / "training.log"
LOG_LEVEL = "INFO"
SAVE_TRAINING_CURVES = True
VERBOSE = True

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_device():
    return DEVICE

def get_checkpoint_path(best=True):
    return BEST_MODEL_PATH if best else LAST_MODEL_PATH

def get_dataset_paths():
    return {
        "train_rgb": TRAIN_RGB_DIR,
        "train_ir": TRAIN_IR_DIR,
        "train_mask": TRAIN_MASK_DIR,
        "val_rgb": VAL_RGB_DIR,
        "val_ir": VAL_IR_DIR,
        "val_mask": VAL_MASK_DIR,
        "test_rgb": TEST_RGB_DIR,
        "test_ir": TEST_IR_DIR,
        "test_mask": TEST_MASK_DIR,
    }

def verify_directories():
    missing = []
    for directory in DIRECTORIES:
        if not directory.exists():
            missing.append(directory)
    if len(missing) == 0:
        print("Directory Verification : PASSED")
        return True
    
    print("Directory Verification : FAILED")
    for directory in missing:
        print(f"Missing: {directory}")
    return False

def verify_dataset():
    dataset_paths = get_dataset_paths()
    missing = []
    for name, folder in dataset_paths.items():
        if not folder.exists():
            missing.append(name)
            
    if len(missing) == 0:
        print("Dataset Verification   : PASSED")
        return True
        
    print("Dataset Verification   : FAILED")
    for folder in missing:
        print(f"Missing Folder: {folder}")
    return False

def print_config():
    print("\n" + "=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)
    print(f"Version            : {VERSION}")
    print(f"Team               : {TEAM_NAME}")
    print(f"Competition        : {COMPETITION}")
    print("\nSYSTEM")
    print("-" * 70)
    print(f"Device             : {DEVICE}")
    print(f"CUDA Available     : {USE_CUDA}")
    print(f"GPU                : {GPU_NAME}")
    print("\nDATASET")
    print("-" * 70)
    print(f"Dataset            : {DATASET_NAME}")
    print(f"Input Size         : {INPUT_SIZE}")
    print(f"Output Size        : {OUTPUT_SIZE}")
    print(f"Num Classes        : {NUM_CLASSES}")
    print(f"Patch Size         : {PATCH_SIZE}")
    print(f"Patch Stride       : {PATCH_STRIDE}")
    print("\nTRAINING")
    print("-" * 70)
    print(f"Epochs             : {EPOCHS}")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Learning Rate      : {LEARNING_RATE}")
    print(f"Optimizer          : {OPTIMIZER}")
    print(f"Scheduler          : {SCHEDULER}")
    print("\nMODEL")
    print("-" * 70)
    print(f"Model              : {MODEL_FULL_NAME}")
    print(f"Input Resolution   : {INPUT_RESOLUTION}")
    print(f"Output Resolution  : {OUTPUT_RESOLUTION}")
    print(f"Upscale Factor     : {UPSCALE_FACTOR}")
    print("=" * 70)

# ============================================================
# CONFIGURATION DICTIONARY
# ============================================================

CONFIG = {
    "project_name": PROJECT_NAME,
    "version": VERSION,
    "model": MODEL_FULL_NAME,
    "device": str(DEVICE),
    "dataset": DATASET_NAME,
    "input_size": INPUT_SIZE,
    "output_size": OUTPUT_SIZE,
    "num_classes": NUM_CLASSES,
    "patch_size": PATCH_SIZE,
    "patch_stride": PATCH_STRIDE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": OPTIMIZER,
    "scheduler": SCHEDULER,
    "checkpoint": str(BEST_MODEL_PATH),
    "results": str(RESULT_DIR),
}

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print_config()
    print()
    verify_directories()
    verify_dataset()
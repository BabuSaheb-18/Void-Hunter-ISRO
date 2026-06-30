"""
==============================================================================
VHNet v2 (Multi-Task)
Dataset Loader

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Loads paired IR, RGB, and Semantic Mask patches.
Input  : 512×512 IR
Target : 1024×1024 RGB + 512x512 Semantic Mask
==============================================================================
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_RGB_DIR,
    TRAIN_IR_DIR,
    TRAIN_MASK_DIR,  # newly added to config
    VAL_RGB_DIR,
    VAL_IR_DIR,
    VAL_MASK_DIR,    # newly added to config
    TEST_RGB_DIR,
    TEST_IR_DIR,
    TEST_MASK_DIR,   # newly added to config
    INPUT_SIZE,
    OUTPUT_SIZE,
)
import cv2
import torch
from torch.utils.data import Dataset

# ============================================================
# DATASET
# ============================================================

class VHNetDataset(Dataset):
    def __init__(self, rgb_dir, ir_dir, mask_dir, training=True):
        self.training = training
        self.rgb_dir = Path(rgb_dir)
        self.ir_dir = Path(ir_dir)
        self.mask_dir = Path(mask_dir)

        # Assuming all images are saved as PNGs
        self.rgb_files = sorted(self.rgb_dir.glob("*.png"))
        self.ir_files = sorted(self.ir_dir.glob("*.png"))
        self.mask_files = sorted(self.mask_dir.glob("*.png"))

        # Strict length validation
        if not (len(self.rgb_files) == len(self.ir_files) == len(self.mask_files)):
            raise RuntimeError(
                f"Dataset count mismatch! "
                f"RGB: {len(self.rgb_files)}, IR: {len(self.ir_files)}, Mask: {len(self.mask_files)}"
            )

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        # 1. Read Images
        rgb = cv2.imread(str(self.rgb_files[index]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        ir = cv2.imread(str(self.ir_files[index]), cv2.IMREAD_GRAYSCALE)
        
        # Load mask as grayscale (assuming pixel values are class indices 0, 1, 2, 3)
        mask = cv2.imread(str(self.mask_files[index]), cv2.IMREAD_GRAYSCALE)

        # 2. Resize
        # IR and RGB use CUBIC for smooth visual interpolation
        ir = cv2.resize(ir, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.resize(rgb, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        
        # MASK MUST use NEAREST to prevent blending categorical class integers
        mask = cv2.resize(mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        # 3. Normalization and Tensor Conversion
        ir = ir.astype("float32") / 255.0
        rgb = rgb.astype("float32") / 255.0
        
        ir = torch.from_numpy(ir).unsqueeze(0)        # Shape: (1, H, W)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # Shape: (3, 2H, 2W)
        
        # Masks are NOT normalized (0 to 1), they stay as long integers for CrossEntropyLoss
        mask = torch.from_numpy(mask).long()          # Shape: (H, W)

        return {
            "ir": ir,
            "rgb": rgb,
            "mask": mask,
        }

# ============================================================
# DATASET FACTORY
# ============================================================

def get_train_dataset():
    return VHNetDataset(
        rgb_dir=TRAIN_RGB_DIR,
        ir_dir=TRAIN_IR_DIR,
        mask_dir=TRAIN_MASK_DIR,
        training=True,
    )

def get_validation_dataset():
    return VHNetDataset(
        rgb_dir=VAL_RGB_DIR,
        ir_dir=VAL_IR_DIR,
        mask_dir=VAL_MASK_DIR,
        training=False,
    )

def get_test_dataset():
    return VHNetDataset(
        rgb_dir=TEST_RGB_DIR,
        ir_dir=TEST_IR_DIR,
        mask_dir=TEST_MASK_DIR,
        training=False,
    )

# ============================================================
# DATASET INFORMATION & VERIFICATION
# ============================================================

def dataset_summary():
    train = get_train_dataset()
    val = get_validation_dataset()
    test = get_test_dataset()

    print("\n" + "=" * 70)
    print("VHNet v2 (Multi-Task) Dataset")
    print("=" * 70)
    print(f"Training Images   : {len(train)}")
    print(f"Validation Images : {len(val)}")
    print(f"Test Images       : {len(test)}")
    print("=" * 70)

def verify_dataset():
    train = get_train_dataset()
    val = get_validation_dataset()
    test = get_test_dataset()

    assert len(train) > 0, "Training dataset is empty."
    assert len(val) > 0, "Validation dataset is empty."
    assert len(test) > 0, "Test dataset is empty."

    sample = train[0]

    assert sample["ir"].shape == (1, INPUT_SIZE, INPUT_SIZE)
    assert sample["rgb"].shape == (3, OUTPUT_SIZE, OUTPUT_SIZE)
    assert sample["mask"].shape == (INPUT_SIZE, INPUT_SIZE)

    print("Dataset Verification : PASSED (IR, RGB, and Mask shapes match specs)")

# ============================================================
# MAIN
# ============================================================

def main():
    dataset_summary()
    verify_dataset()

    sample = get_train_dataset()[0]
    
    print("\n" + "=" * 70)
    print("Sample Shapes Output")
    print("=" * 70)
    print(f"IR   (Input)       : {sample['ir'].shape}")
    print(f"RGB  (Target HR)   : {sample['rgb'].shape}")
    print(f"Mask (Target Seg)  : {sample['mask'].shape}")
    print("=" * 70)

if __name__ == "__main__":
    main()
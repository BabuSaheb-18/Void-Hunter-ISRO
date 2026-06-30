"""
==============================================================================
VHNet v2 (Multi-Task)
Evaluation Script

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Computes:
✔ PSNR  (Image Quality)
✔ SSIM  (Structural Integrity)
✔ MSE   (Reconstruction Error)
✔ LPIPS (Perceptual Realism)
✔ mIoU  (Semantic Segmentation Accuracy)
==============================================================================
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from dataset import get_test_dataset
from model import VHNet_v2 # Updated to v2
from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    BEST_MODEL_PATH,
    NUM_CLASSES, # Imported for mIoU calculation
)

# ============================================================
# DATASET
# ============================================================

test_dataset = get_test_dataset()

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# ============================================================
# MODEL
# ============================================================

model = VHNet_v2(num_classes=NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ============================================================
# LPIPS
# ============================================================

if LPIPS_AVAILABLE:
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
else:
    lpips_model = None

print("=" * 70)
print("VHNet v2 (Multi-Task) Evaluation")
print("=" * 70)
print("Test Images :", len(test_dataset))
print("=" * 70)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_psnr(prediction, target):
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))
    return peak_signal_noise_ratio(target, prediction, data_range=1.0)

def calculate_ssim(prediction, target):
    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))
    target = np.transpose(target, (1, 2, 0))
    return structural_similarity(target, prediction, channel_axis=2, data_range=1.0)

def calculate_mse(prediction, target):
    return torch.mean((prediction - target) ** 2).item()

def calculate_lpips(prediction, target):
    if lpips_model is None:
        return 0.0
    prediction = prediction * 2.0 - 1.0
    target = target * 2.0 - 1.0
    return lpips_model(prediction, target).mean().item()

def calculate_miou(prediction_logits, target_mask, num_classes):
    """
    Calculates Mean Intersection over Union (mIoU) for the Semantic Mask.
    prediction_logits: Tensor of shape (C, H, W)
    target_mask: Tensor of shape (H, W) containing integer class labels
    """
    # Convert logits to predicted class labels by taking the argmax across the channels
    pred_mask = torch.argmax(prediction_logits, dim=0) # Shape: (H, W)
    
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (target_mask == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            # If the class isn't in the prediction OR the ground truth, skip it
            continue
            
        ious.append(intersection / union)
        
    return np.mean(ious) if ious else 0.0

# ============================================================
# EVALUATION LOOP
# ============================================================

@torch.no_grad()
def evaluate():
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    lpips_scores = []
    miou_scores = [] # Added for segmentation

    for batch in test_loader:
        ir = batch["ir"].to(DEVICE)
        rgb_gt = batch["rgb"].to(DEVICE)
        mask_gt = batch["mask"].to(DEVICE)

        # Unpack the dual outputs
        rgb_pred, mask_pred = model(ir)

        for i in range(rgb_pred.size(0)):
            r_pred = rgb_pred[i]
            r_gt = rgb_gt[i]
            m_pred = mask_pred[i]
            m_gt = mask_gt[i]

            # 1. RGB Metrics
            psnr_scores.append(calculate_psnr(r_pred, r_gt))
            ssim_scores.append(calculate_ssim(r_pred, r_gt))
            mse_scores.append(calculate_mse(r_pred, r_gt))
            lpips_scores.append(
                calculate_lpips(r_pred.unsqueeze(0), r_gt.unsqueeze(0))
            )
            
            # 2. Semantic Mask Metric
            miou_scores.append(calculate_miou(m_pred, m_gt, NUM_CLASSES))

    return {
        "PSNR": np.mean(psnr_scores),
        "SSIM": np.mean(ssim_scores),
        "MSE": np.mean(mse_scores),
        "LPIPS": np.mean(lpips_scores),
        "mIoU": np.mean(miou_scores),
    }

# ============================================================
# RESULTS PRINTER
# ============================================================

def print_results(results):
    print("\n" + "=" * 70)
    print("VHNet v2 Evaluation Results")
    print("=" * 70)
    print("[ RGB Reconstruction ]")
    print(f"PSNR   : {results['PSNR']:.4f} dB")
    print(f"SSIM   : {results['SSIM']:.4f}")
    print(f"MSE    : {results['MSE']:.8f}")
    print(f"LPIPS  : {results['LPIPS']:.6f}")
    print("-" * 70)
    print("[ Object Interpretation ]")
    print(f"mIoU   : {results['mIoU']:.4f} (Higher is better, max 1.0)")
    print("=" * 70)

# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    output_file = BEST_MODEL_PATH.parent / "evaluation_v2.txt"

    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VHNet v2 (Multi-Task) Evaluation\n")
        f.write("=" * 70 + "\n\n")
        f.write("[ RGB Reconstruction ]\n")
        f.write(f"PSNR   : {results['PSNR']:.4f} dB\n")
        f.write(f"SSIM   : {results['SSIM']:.4f}\n")
        f.write(f"MSE    : {results['MSE']:.8f}\n")
        f.write(f"LPIPS  : {results['LPIPS']:.6f}\n\n")
        f.write("[ Object Interpretation ]\n")
        f.write(f"mIoU   : {results['mIoU']:.4f}\n")

    print(f"\nResults saved to -> {output_file}")

# ============================================================
# MAIN
# ============================================================

def main():
    results = evaluate()
    print_results(results)
    save_results(results)

if __name__ == "__main__":
    main()
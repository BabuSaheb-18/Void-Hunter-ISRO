import os
import json

import cv2
import lpips
import torch
import numpy as np

from tqdm.auto import tqdm

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from dataset import LandsatDataset
from model import IR2RGBUNet


# ==========================================================
# Configuration
# ==========================================================

TEST_RGB = "/content/drive/MyDrive/VoidHunterDataset/test/rgb"

TEST_IR = "/content/drive/MyDrive/VoidHunterDataset/test/ir"

SAVE_DIR = "/content/drive/MyDrive/VoidHunterModel"

BEST_MODEL = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

EVALUATION_FILE = os.path.join(
    SAVE_DIR,
    "evaluation.json"
)

SAMPLE_DIR = os.path.join(
    SAVE_DIR,
    "sample_predictions"
)

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True
)


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print()

print("=" * 70)

print("VOID HUNTER MODEL EVALUATION")

print("=" * 70)

print("Device :", DEVICE)

if torch.cuda.is_available():

    print("GPU :", torch.cuda.get_device_name(0))

else:

    print("GPU : CPU")

print("=" * 70)

print()


# ==========================================================
# Dataset
# ==========================================================

test_dataset = LandsatDataset(

    rgb_dir=TEST_RGB,

    ir_dir=TEST_IR

)

print(f"Test Images : {len(test_dataset)}")

print()


# ==========================================================
# Load Model
# ==========================================================

model = IR2RGBUNet().to(DEVICE)

checkpoint = torch.load(

    BEST_MODEL,

    map_location=DEVICE

)

if "model_state_dict" in checkpoint:

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

else:

    model.load_state_dict(

        checkpoint

    )

model.eval()


# ==========================================================
# LPIPS
# ==========================================================

lpips_model = lpips.LPIPS(

    net="alex"

).to(DEVICE)

lpips_model.eval()


# ==========================================================
# Metric Lists
# ==========================================================

psnr_scores = []

ssim_scores = []

mse_scores = []

lpips_scores = []

print("Model Loaded Successfully")

print()
# ==========================================================
# Evaluation
# ==========================================================

print("=" * 70)
print("STARTING EVALUATION")
print("=" * 70)

with torch.no_grad():

    for index in tqdm(

        range(len(test_dataset)),

        desc="Evaluating"

    ):

        ir, gt = test_dataset[index]

        ir_tensor = ir.unsqueeze(0).to(DEVICE)

        gt_tensor = gt.unsqueeze(0).to(DEVICE)

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model(ir_tensor)

        pred = prediction.squeeze(0).permute(
            1,
            2,
            0
        ).cpu().numpy()

        gt_img = gt.squeeze(0).permute(
            1,
            2,
            0
        ).numpy()

        pred = np.clip(
            pred,
            0,
            1
        )

        gt_img = np.clip(
            gt_img,
            0,
            1
        )

        # --------------------------------------------------
        # MSE
        # --------------------------------------------------

        mse = np.mean(
            (pred - gt_img) ** 2
        )

        mse_scores.append(
            float(mse)
        )

        # --------------------------------------------------
        # PSNR
        # --------------------------------------------------

        psnr = peak_signal_noise_ratio(

            gt_img,

            pred,

            data_range=1.0

        )

        psnr_scores.append(
            float(psnr)
        )

        # --------------------------------------------------
        # SSIM
        # --------------------------------------------------

        ssim = structural_similarity(

            gt_img,

            pred,

            channel_axis=2,

            data_range=1.0

        )

        ssim_scores.append(
            float(ssim)
        )

        # --------------------------------------------------
        # LPIPS
        # --------------------------------------------------

        gt_lpips = (

            gt_tensor * 2

        ) - 1

        pred_lpips = (

            prediction * 2

        ) - 1

        lpips_value = lpips_model(

            gt_lpips,

            pred_lpips

        )

        lpips_scores.append(

            float(

                lpips_value.item()

            )

        )

        # --------------------------------------------------
        # Save First 10 Samples
        # --------------------------------------------------

        if index < 10:

            ir_image = (

                ir.squeeze()

                .numpy()

                * 255

            ).astype(np.uint8)

            pred_image = (

                pred * 255

            ).astype(np.uint8)

            gt_image = (

                gt_img * 255

            ).astype(np.uint8)

            pred_image = cv2.cvtColor(

                pred_image,

                cv2.COLOR_RGB2BGR

            )

            gt_image = cv2.cvtColor(

                gt_image,

                cv2.COLOR_RGB2BGR

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{index+1:02d}_ir.png"

                ),

                ir_image

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{index+1:02d}_prediction.png"

                ),

                pred_image

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{index+1:02d}_groundtruth.png"

                ),

                gt_image

            )

print()

print("=" * 70)
print("EVALUATION FINISHED")
print("=" * 70)
print()
# ==========================================================
# Evaluation Loop
# ==========================================================

print("=" * 70)
print("STARTING MODEL EVALUATION")
print("=" * 70)

with torch.no_grad():

    for idx in tqdm(

        range(len(test_dataset)),

        desc="Evaluating"

    ):

        # --------------------------------------------------
        # Load Sample
        # --------------------------------------------------

        ir, gt = test_dataset[idx]

        ir_tensor = ir.unsqueeze(0).to(DEVICE)

        gt_tensor = gt.unsqueeze(0).to(DEVICE)

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model(ir_tensor)

        pred = prediction.squeeze(0).permute(
            1,
            2,
            0
        ).cpu().numpy()

        gt_img = gt.permute(
            1,
            2,
            0
        ).cpu().numpy()

        pred = np.clip(
            pred,
            0,
            1
        )

        gt_img = np.clip(
            gt_img,
            0,
            1
        )

        # --------------------------------------------------
        # MSE
        # --------------------------------------------------

        mse = np.mean(

            (pred - gt_img) ** 2

        )

        mse_scores.append(float(mse))

        # --------------------------------------------------
        # PSNR
        # --------------------------------------------------

        psnr = peak_signal_noise_ratio(

            gt_img,

            pred,

            data_range=1.0

        )

        psnr_scores.append(float(psnr))

        # --------------------------------------------------
        # SSIM
        # --------------------------------------------------

        ssim = structural_similarity(

            gt_img,

            pred,

            channel_axis=2,

            data_range=1.0

        )

        ssim_scores.append(float(ssim))

        # --------------------------------------------------
        # LPIPS
        # --------------------------------------------------

        gt_lpips = gt_tensor * 2 - 1

        pred_lpips = prediction * 2 - 1

        lpips_value = lpips_model(

            gt_lpips,

            pred_lpips

        )

        lpips_scores.append(

            float(lpips_value.item())

        )

        # --------------------------------------------------
        # Save Sample Predictions
        # --------------------------------------------------

        if idx < 10:

            ir_image = (

                ir.squeeze().cpu().numpy()

                * 255

            ).astype(np.uint8)

            pred_image = (

                pred * 255

            ).astype(np.uint8)

            gt_image = (

                gt_img * 255

            ).astype(np.uint8)

            pred_image = cv2.cvtColor(

                pred_image,

                cv2.COLOR_RGB2BGR

            )

            gt_image = cv2.cvtColor(

                gt_image,

                cv2.COLOR_RGB2BGR

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{idx+1:02d}_ir.png"

                ),

                ir_image

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{idx+1:02d}_prediction.png"

                ),

                pred_image

            )

            cv2.imwrite(

                os.path.join(

                    SAMPLE_DIR,

                    f"{idx+1:02d}_groundtruth.png"

                ),

                gt_image

            )

print()

print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
print()
# ==========================================================
# Calculate Average Metrics
# ==========================================================

average_psnr = float(np.mean(psnr_scores))

average_ssim = float(np.mean(ssim_scores))

average_mse = float(np.mean(mse_scores))

average_lpips = float(np.mean(lpips_scores))


# ==========================================================
# Save Evaluation Results
# ==========================================================

evaluation = {

    "test_images": len(test_dataset),

    "psnr": round(average_psnr, 4),

    "ssim": round(average_ssim, 4),

    "mse": round(average_mse, 6),

    "lpips": round(average_lpips, 6),

    "device": str(DEVICE),

    "framework": f"PyTorch {torch.__version__}"

}

with open(

    EVALUATION_FILE,

    "w"

) as f:

    json.dump(

        evaluation,

        f,

        indent=4

    )


# ==========================================================
# Evaluation Report
# ==========================================================

print()

print("=" * 70)

print("VOID HUNTER MODEL EVALUATION")

print("=" * 70)

print(f"Test Images        : {len(test_dataset)}")

print()

print("Evaluation Metrics")

print("-" * 70)

print(f"PSNR   : {average_psnr:.4f} dB")

print(f"SSIM   : {average_ssim:.4f}")

print(f"MSE    : {average_mse:.6f}")

print(f"LPIPS  : {average_lpips:.6f}")

print("-" * 70)

print()

print("Files Generated")

print("-" * 70)

print(EVALUATION_FILE)

print(SAMPLE_DIR)

print("-" * 70)

print()

print("Evaluation Completed Successfully.")

print("=" * 70)
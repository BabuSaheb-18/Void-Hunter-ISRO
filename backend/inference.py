import os
import sys
import time

import cv2
import numpy as np
import torch

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import lpips

# ==========================================================
# Project Path
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

from models.model import IR2RGBUNet


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)


# ==========================================================
# Load Model
# ==========================================================

model = IR2RGBUNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()


# ==========================================================
# Lazy LPIPS Loader
# ==========================================================

lpips_model = None

def get_lpips_model():
    global lpips_model

    if lpips_model is None:
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
        lpips_model.eval()

    return lpips_model


# ==========================================================
# Count Parameters
# ==========================================================

TOTAL_PARAMETERS = sum(
    p.numel()
    for p in model.parameters()
)


# ==========================================================
# Training Statistics
# ==========================================================

TRAINING_LOSS = "N/A"
VALIDATION_LOSS = "N/A"
TRAINED_EPOCH = "N/A"

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        TRAINING_LOSS = checkpoint.get(
            "training_loss",
            "N/A"
        )

        VALIDATION_LOSS = checkpoint.get(
            "validation_loss",
            "N/A"
        )

        TRAINED_EPOCH = checkpoint.get(
            "epoch",
            "N/A"
        )

except Exception as e:

    print("Unable to load checkpoint information:", e)

# ==========================================================
# Startup Log
# ==========================================================

print("=" * 50)
print("VOID HUNTER")
print("=" * 50)
print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Parameters :", f"{TOTAL_PARAMETERS:,}")
print("Epoch :", TRAINED_EPOCH)
print("Training Loss :", TRAINING_LOSS)
print("Validation Loss :", VALIDATION_LOSS)
print("=" * 50)
# ==========================================================
# Inference
# ==========================================================

def enhance_infrared(input_path, output_path):
    print(">>> enhance_infrared() called")
    print("Input:", input_path)
    print("Output:", output_path)

    start_time = time.time()

    # ------------------------------------------------------
    # Read Input Image
    # ------------------------------------------------------

    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise Exception("Unable to read input image.")

    filename = os.path.basename(input_path)

    input_height, input_width = image.shape

    input_file_size = round(
        os.path.getsize(input_path) / 1024,
        2
    )

    # ------------------------------------------------------
    # Prepare Model Input
    # ------------------------------------------------------

    resized = cv2.resize(
        image,
        (256, 256)
    )

    normalized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # ------------------------------------------------------
    # AI Inference
    # ------------------------------------------------------

    with torch.no_grad():

        prediction = model(tensor)

    prediction = prediction.squeeze(0)

    prediction = prediction.permute(
        1,
        2,
        0
    )

    prediction = prediction.cpu().numpy()

    prediction = np.clip(
        prediction,
        0,
        1
    )

    prediction = (
        prediction * 255
    ).astype(np.uint8)

    prediction = cv2.cvtColor(
        prediction,
        cv2.COLOR_RGB2BGR
    )

    # ------------------------------------------------------
    # Save Enhanced Image
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        prediction
    )

    output_height, output_width = prediction.shape[:2]

    output_file_size = round(
        os.path.getsize(output_path) / 1024,
        2
    )

    # ------------------------------------------------------
    # Ground Truth Detection
    # ------------------------------------------------------

    filename = os.path.basename(input_path)

    # Convert IR filename to RGB filename
    gt_filename = filename.replace(
        "ir_",
        "rgb_",
        1
    )

    # ------------------------------------------------------
    # Search Ground Truth in all dataset splits
    # ------------------------------------------------------

    gt_path = None

    search_locations = [

        os.path.join(BASE_DIR, "dataset", "test", "rgb"),

        os.path.join(BASE_DIR, "dataset", "val", "rgb"),

        os.path.join(BASE_DIR, "dataset", "train", "rgb")

    ]

    for folder in search_locations:

        candidate = os.path.join(folder, gt_filename)

        if os.path.exists(candidate):

            gt_path = candidate

            print(f"Ground Truth Found : {candidate}")

            break

    if gt_path is None:

        print("Ground Truth Not Found")
    psnr = "N/A"
    ssim = "N/A"
    mse = "N/A"
    lpips_score = "N/A"
    print("Looking for:", gt_path)
    print("Found:", os.path.exists(gt_path))

    if gt_path is not None:

        gt = cv2.imread(gt_path)

        if gt is not None:

            gt = cv2.resize(
                gt,
                (256, 256)
            )

            pred = cv2.imread(
                output_path
            )

            gt_rgb = cv2.cvtColor(
                gt,
                cv2.COLOR_BGR2RGB
            )

            pred_rgb = cv2.cvtColor(
                pred,
                cv2.COLOR_BGR2RGB
            )

            gt_float = gt_rgb.astype(
                np.float32
            ) / 255.0

            pred_float = pred_rgb.astype(
                np.float32
            ) / 255.0

            mse = round(
                np.mean(
                    (gt_float - pred_float) ** 2
                ),
                6
            )

            psnr = round(
                peak_signal_noise_ratio(
                    gt_float,
                    pred_float,
                    data_range=1.0
                ),
                2
            )

            ssim = round(
                structural_similarity(
                    gt_float,
                    pred_float,
                    channel_axis=2,
                    data_range=1.0
                ),
                4
            )

            gt_tensor = torch.from_numpy(
                gt_float
            ).permute(
                2,
                0,
                1
            ).unsqueeze(0)

            pred_tensor = torch.from_numpy(
                pred_float
            ).permute(
                2,
                0,
                1
            ).unsqueeze(0)

            gt_tensor = (
                gt_tensor.to(DEVICE) * 2
            ) - 1

            pred_tensor = (
                pred_tensor.to(DEVICE) * 2
            ) - 1

            with torch.no_grad():

                lpips_score = round(
                    get_lpips_model()(
                        gt_tensor,
                        pred_tensor
                    ).item(),
                    4
                )

    # ------------------------------------------------------
    # Timing
    # ------------------------------------------------------

    inference_time = round(
        time.time() - start_time,
        3
    )

    # ------------------------------------------------------
    # Continue to Part 3
    # ------------------------------------------------------
        # ------------------------------------------------------
    # Final Report
    # ------------------------------------------------------

    report = {

        # Status

        "status": "SUCCESS",

        # Images

        "output_path": output_path,

        "filename": filename,

        # Performance

        "inference_time": f"{inference_time:.3f} sec",

        "device": str(DEVICE).upper(),

        # Resolution

        "input_resolution":
            f"{input_width} × {input_height}",

        "output_resolution":
            f"{output_width} × {output_height}",

        # File Size

        "input_size":
            f"{input_file_size} KB",

        "output_size":
            f"{output_file_size} KB",

        # Model

        "model_name":
            "Void Hunter U-Net",

        "model_version":
            "v2.0",

        "framework":
            f"PyTorch {torch.__version__}",

        "prediction":
            "Infrared → RGB",

        "parameters":
            f"{TOTAL_PARAMETERS:,}",

        # Evaluation Metrics

        "psnr": psnr,

        "ssim": ssim,

        "mse": mse,

        "lpips": lpips_score,

        # Training Statistics

        "training_loss": TRAINING_LOSS,

        "validation_loss": VALIDATION_LOSS,

        # Ground Truth

        "ground_truth_found":
             gt_path is not None

    }

    return report
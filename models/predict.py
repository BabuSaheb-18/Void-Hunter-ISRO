import os
import cv2
import torch

from model import IR2RGBUNet


# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "models/best_model.pth"

INPUT_IMAGE = "dataset/train/ir/ir_000001.png"

OUTPUT_IMAGE = "output/predicted_rgb.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Create Output Folder
# ==========================================================

os.makedirs("output", exist_ok=True)


# ==========================================================
# Load Model
# ==========================================================

model = IR2RGBUNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("✅ Model Loaded")


# ==========================================================
# Read Image
# ==========================================================

image = cv2.imread(
    INPUT_IMAGE,
    cv2.IMREAD_GRAYSCALE
)

image = cv2.resize(image, (256, 256))

image = image.astype("float32") / 255.0

image = torch.from_numpy(image)

image = image.unsqueeze(0).unsqueeze(0)

image = image.to(DEVICE)


# ==========================================================
# Prediction
# ==========================================================

with torch.no_grad():

    prediction = model(image)


prediction = prediction.squeeze(0)

prediction = prediction.permute(1, 2, 0)

prediction = prediction.cpu().numpy()

prediction = (prediction * 255).astype("uint8")


# ==========================================================
# Save Result
# ==========================================================

prediction = cv2.cvtColor(
    prediction,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    OUTPUT_IMAGE,
    prediction
)

print("====================================")
print("Prediction Completed Successfully!")
print("Saved :", OUTPUT_IMAGE)
print("====================================")
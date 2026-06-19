import os
import time
import torch

from torch.utils.data import DataLoader

from dataset import LandsatDataset
from model import IR2RGBUNet
from loss import CombinedLoss


# ==========================================================
# Configuration
# ==========================================================

TRAIN_RGB = "dataset/train/rgb"
TRAIN_IR = "dataset/train/ir"

VAL_RGB = "dataset/val/rgb"
VAL_IR = "dataset/val/ir"

BATCH_SIZE = 8
EPOCHS = 10          # First test (increase to 50 later)
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_PATH = "models/best_model.pth"


# ==========================================================
# Device Information
# ==========================================================

print("=" * 60)
print("Infrared → RGB Training")
print("=" * 60)
print("Device :", DEVICE)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
else:
    print("GPU    : Not Available (Training on CPU)")

print("=" * 60)


# ==========================================================
# Dataset
# ==========================================================

train_dataset = LandsatDataset(
    rgb_dir=TRAIN_RGB,
    ir_dir=TRAIN_IR
)

val_dataset = LandsatDataset(
    rgb_dir=VAL_RGB,
    ir_dir=VAL_IR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Training Batches  : {len(train_loader)}")
print(f"Validation Batches: {len(val_loader)}")
print("=" * 60)


# ==========================================================
# Model
# ==========================================================

model = IR2RGBUNet().to(DEVICE)

criterion = CombinedLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_val_loss = float("inf")


# ==========================================================
# Training
# ==========================================================

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    start_time = time.time()

    model.train()

    running_train_loss = 0.0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for batch_idx, (ir, rgb) in enumerate(train_loader):

        ir = ir.to(DEVICE)
        rgb = rgb.to(DEVICE)

        optimizer.zero_grad()

        prediction = model(ir)

        loss = criterion(prediction, rgb)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):

            print(
                f" Batch [{batch_idx+1:02d}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_train_loss / len(train_loader)


    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for ir, rgb in val_loader:

            ir = ir.to(DEVICE)
            rgb = rgb.to(DEVICE)

            prediction = model(ir)

            loss = criterion(prediction, rgb)

            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)


    # ======================================================
    # Save Best Model
    # ======================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "training_loss": train_loss,
            "validation_loss": val_loss
        }, SAVE_PATH)

        print("✅ Best model saved!")


    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Epoch Time : {elapsed:.2f} sec")
    print("-" * 60)


print("\n" + "=" * 60)
print("Training Completed Successfully!")
print("=" * 60)
print(f"Best Validation Loss : {best_val_loss:.4f}")
print(f"Model Saved At       : {SAVE_PATH}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)
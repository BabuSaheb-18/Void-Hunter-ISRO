import os
import json
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torch.amp import autocast
from torch.amp import GradScaler

from tqdm.auto import tqdm

from dataset import LandsatDataset
from model import IR2RGBUNet
from loss import CombinedLoss


# ==========================================================
# Configuration
# ==========================================================

TRAIN_RGB = "/content/drive/MyDrive/VoidHunterDataset/train/rgb"
TRAIN_IR = "/content/drive/MyDrive/VoidHunterDataset/train/ir"

VAL_RGB = "/content/drive/MyDrive/VoidHunterDataset/val/rgb"
VAL_IR = "/content/drive/MyDrive/VoidHunterDataset/val/ir"

SAVE_DIR = "/content/drive/MyDrive/VoidHunterModel"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

BEST_MODEL = os.path.join(
    SAVE_DIR,
    "best_model.pth"
)

LAST_CHECKPOINT = os.path.join(
    SAVE_DIR,
    "last_checkpoint.pth"
)

METRICS_FILE = os.path.join(
    SAVE_DIR,
    "metrics.json"
)

HISTORY_FILE = os.path.join(
    SAVE_DIR,
    "history.json"
)

LOSS_CURVE = os.path.join(
    SAVE_DIR,
    "loss_curve.png"
)

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

PATIENCE = 10

SEED = 42


# ==========================================================
# Reproducibility
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

cudnn.benchmark = True


# ==========================================================
# GPU Information
# ==========================================================

print()

print("=" * 70)

print("VOID HUNTER AI TRAINING")

print("=" * 70)

print("PyTorch :", torch.__version__)

print("Device  :", DEVICE)

if torch.cuda.is_available():

    print("GPU     :", torch.cuda.get_device_name(0))

else:

    print("GPU     : CPU")

print("=" * 70)

print()


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


# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=2,

    pin_memory=True,

    persistent_workers=True,

    drop_last=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=2,

    pin_memory=True,

    persistent_workers=True

)


print("=" * 70)

print("DATASET")

print("=" * 70)

print(f"Training Images   : {len(train_dataset)}")

print(f"Validation Images : {len(val_dataset)}")

print(f"Training Batches  : {len(train_loader)}")

print(f"Validation Batches: {len(val_loader)}")

print("=" * 70)

print()


# ==========================================================
# Model
# ==========================================================

model = IR2RGBUNet().to(DEVICE)

criterion = CombinedLoss()

optimizer = AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3

)

scaler = GradScaler("cuda")


# ==========================================================
# Statistics
# ==========================================================

TOTAL_PARAMETERS = sum(

    p.numel()

    for p in model.parameters()

)

print(f"Model Parameters : {TOTAL_PARAMETERS:,}")

print()


# ==========================================================
# Resume Training
# ==========================================================

start_epoch = 0

best_val_loss = float("inf")

train_history = []

val_history = []

if os.path.exists(LAST_CHECKPOINT):

    print("Resuming previous training...\n")

    checkpoint = torch.load(

        LAST_CHECKPOINT,

        map_location=DEVICE

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    scheduler.load_state_dict(

        checkpoint["scheduler_state_dict"]

    )

    scaler.load_state_dict(

        checkpoint["scaler_state_dict"]

    )

    start_epoch = checkpoint["epoch"]

    best_val_loss = checkpoint["best_val_loss"]

    train_history = checkpoint.get(

        "train_history",

        []

    )

    val_history = checkpoint.get(

        "val_history",

        []

    )

    print(f"Continuing from Epoch {start_epoch}")

else:

    print("Starting New Training\n")


# ==========================================================
# Early Stopping
# ==========================================================

early_stop_counter = 0
# ==========================================================
# Training
# ==========================================================

print("=" * 70)
print("TRAINING STARTED")
print("=" * 70)

for epoch in range(start_epoch, EPOCHS):

    print()
    print("=" * 70)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 70)

    epoch_start = time.time()

    model.train()

    running_train_loss = 0.0

    train_bar = tqdm(

        train_loader,

        desc="Training",

        leave=False

    )

    for ir, rgb in train_bar:

        ir = ir.to(

            DEVICE,

            non_blocking=True

        )

        rgb = rgb.to(

            DEVICE,

            non_blocking=True

        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):

            prediction = model(ir)

            loss = criterion(

                prediction,

                rgb

            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_train_loss += loss.item()

        train_bar.set_postfix(

            Loss=f"{loss.item():.4f}"

        )

    train_loss = running_train_loss / len(train_loader)

    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    running_val_loss = 0.0

    val_bar = tqdm(

        val_loader,

        desc="Validation",

        leave=False

    )

    with torch.no_grad():

        for ir, rgb in val_bar:

            ir = ir.to(

                DEVICE,

                non_blocking=True

            )

            rgb = rgb.to(

                DEVICE,

                non_blocking=True

            )

            with autocast(device_type="cuda"):

                prediction = model(ir)

                loss = criterion(

                    prediction,

                    rgb

                )

            running_val_loss += loss.item()

            val_bar.set_postfix(

                Loss=f"{loss.item():.4f}"

            )

    val_loss = running_val_loss / len(val_loader)

    scheduler.step(val_loss)

    train_history.append(train_loss)

    val_history.append(val_loss)

    epoch_time = time.time() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]

    print()

    print("-" * 70)

    print(f"Training Loss   : {train_loss:.6f}")

    print(f"Validation Loss : {val_loss:.6f}")

    print(f"Learning Rate   : {current_lr:.8f}")

    print(f"Epoch Time      : {epoch_time:.2f} sec")

    print("-" * 70)

    # ======================================================
    # Save Latest Checkpoint
    # ======================================================

    checkpoint = {

        "epoch": epoch + 1,

        "best_val_loss": best_val_loss,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "train_history": train_history,

        "val_history": val_history

    }

    torch.save(

        checkpoint,

        LAST_CHECKPOINT

    )

    # ======================================================
    # Save Best Model
    # ======================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(

            checkpoint,

            BEST_MODEL

        )

        print()

        print("Best Model Updated")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

        print()

        print(

            f"No Improvement ({early_stop_counter}/{PATIENCE})"

        )

    # ======================================================
    # Save Metrics
    # ======================================================

    metrics = {

        "epoch": epoch + 1,

        "training_loss": float(train_loss),

        "validation_loss": float(val_loss),

        "learning_rate": float(current_lr),

        "best_validation_loss": float(best_val_loss),

        "total_parameters": int(TOTAL_PARAMETERS)

    }

    with open(

        METRICS_FILE,

        "w"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4

        )

    # ======================================================
    # Save History
    # ======================================================

    history = {

        "train_loss": train_history,

        "validation_loss": val_history

    }

    with open(

        HISTORY_FILE,

        "w"

    ) as f:

        json.dump(

            history,

            f,

            indent=4

        )

    # ======================================================
    # Early Stopping
    # ======================================================

    if early_stop_counter >= PATIENCE:

        print()

        print("=" * 70)

        print("EARLY STOPPING TRIGGERED")

        print("=" * 70)

        break

    torch.cuda.empty_cache()
    # ==========================================================
# Generate Loss Curve
# ==========================================================

print()

print("=" * 70)

print("Generating Loss Curve...")

print("=" * 70)

plt.figure(figsize=(10,6))

plt.plot(

    train_history,

    label="Training Loss",

    linewidth=2

)

plt.plot(

    val_history,

    label="Validation Loss",

    linewidth=2

)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Void Hunter Training History")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(

    LOSS_CURVE,

    dpi=300

)

plt.close()

print("Loss curve saved.")

print()


# ==========================================================
# Save Final Metrics
# ==========================================================

final_metrics = {

    "epochs_completed": len(train_history),

    "training_loss": float(train_history[-1]),

    "validation_loss": float(val_history[-1]),

    "best_validation_loss": float(best_val_loss),

    "learning_rate": float(

        optimizer.param_groups[0]["lr"]

    ),

    "batch_size": BATCH_SIZE,

    "total_parameters": int(TOTAL_PARAMETERS),

    "device": str(DEVICE),

    "framework": f"PyTorch {torch.__version__}"

}

with open(

    METRICS_FILE,

    "w"

) as f:

    json.dump(

        final_metrics,

        f,

        indent=4

    )


# ==========================================================
# Save Training History
# ==========================================================

history = {

    "train_loss": train_history,

    "validation_loss": val_history

}

with open(

    HISTORY_FILE,

    "w"

) as f:

    json.dump(

        history,

        f,

        indent=4

    )


# ==========================================================
# Training Summary
# ==========================================================

print()

print("=" * 70)

print("VOID HUNTER AI TRAINING COMPLETED")

print("=" * 70)

print(f"Epochs Completed      : {len(train_history)}")

print(f"Best Validation Loss  : {best_val_loss:.6f}")

print(f"Final Training Loss   : {train_history[-1]:.6f}")

print(f"Final Validation Loss : {val_history[-1]:.6f}")

print(f"Learning Rate         : {optimizer.param_groups[0]['lr']:.8f}")

print(f"Batch Size            : {BATCH_SIZE}")

print(f"Parameters            : {TOTAL_PARAMETERS:,}")

print()

print("Files Generated")

print("-" * 70)

print(BEST_MODEL)

print(LAST_CHECKPOINT)

print(METRICS_FILE)

print(HISTORY_FILE)

print(LOSS_CURVE)

print("-" * 70)

print()

print("Training Finished Successfully.")

print("=" * 70)
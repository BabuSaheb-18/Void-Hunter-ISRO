"""
============================================================
VHNet v2 (Multi-Task)
Training Script

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Input  : IR 
Output : 2x RGB + 1x Semantic Mask
Google Colab Optimized (Includes Plateau Breaker & Data Shield)
============================================================
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Assuming your dataset yields dicts with "ir", "rgb", and "mask"
from dataset import get_train_dataset, get_validation_dataset
from model import VHNet_v2
from loss import build_loss

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE
)

# ============================================================
# DATASET
# ============================================================
train_dataset = get_train_dataset()
val_dataset = get_validation_dataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False,
)

# ============================================================
# MODEL SETUP
# ============================================================
model = VHNet_v2(num_classes=4).to(DEVICE)
criterion = build_loss().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Aggressive scheduler to break through training plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
)

# Initialize Scaler specifically for CUDA (Modern PyTorch 2.x syntax)
scaler = GradScaler('cuda')

best_loss = float("inf")
start_epoch = 1

if BEST_MODEL_PATH.exists():
    print("=" * 70)
    print("Loading Checkpoint")
    print("=" * 70)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_loss = checkpoint["best_loss"]
    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming From Epoch {start_epoch}")

print("=" * 70)
print("VHNet v2 (Multi-Task) Training")
print("=" * 70)
print("Device             :", DEVICE)
print("Train Images       :", len(train_dataset))
print("Validation Images  :", len(val_dataset))
print("Train Batches      :", len(train_loader))
print("Validation Batches :", len(val_loader))
print("=" * 70)

# ============================================================
# TRAINING EPOCH
# ============================================================
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        ir = batch["ir"].to(DEVICE, non_blocking=True)
        rgb_gt = batch["rgb"].to(DEVICE, non_blocking=True)
        mask_gt = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass within autocast context for CUDA
        with autocast('cuda'):
            rgb_pred, mask_pred = model(ir)
            losses = criterion(rgb_pred, rgb_gt, mask_pred, mask_gt)
            loss = losses["loss"]

        # ==========================================
        # DATA SHIELD: Skip corrupted batches (NaN/Inf)
        # ==========================================
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[WARNING] Corrupted batch {batch_idx+1} skipped to protect the model.")
            continue
            
        # ==========================================
        # MODERN BACKPROPAGATION & CLIPPING
        # ==========================================
        scaler.scale(loss).backward()
        
        # Unscale before clipping to get actual gradient values
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step and update
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        valid_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss : {loss.item():.6f}"
            )

    # Avoid division by zero if entire epoch is skipped (highly unlikely)
    epoch_loss = running_loss / valid_batches if valid_batches > 0 else float("inf")
    elapsed = time.time() - start_time

    return epoch_loss, elapsed

# ============================================================
# VALIDATION
# ============================================================
@torch.no_grad()
def validate():
    model.eval()
    running_loss = 0.0
    valid_batches = 0

    for batch in val_loader:
        ir = batch["ir"].to(DEVICE, non_blocking=True)
        rgb_gt = batch["rgb"].to(DEVICE, non_blocking=True)
        mask_gt = batch["mask"].to(DEVICE, non_blocking=True)

        with autocast('cuda'):
            rgb_pred, mask_pred = model(ir)
            losses = criterion(rgb_pred, rgb_gt, mask_pred, mask_gt)
            loss = losses["loss"]
            
        if not (torch.isnan(loss) or torch.isinf(loss)):
            running_loss += loss.item()
            valid_batches += 1

    return running_loss / valid_batches if valid_batches > 0 else float("inf")

# ============================================================
# TRAIN LOOP
# ============================================================
def train():
    global best_loss

    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70)

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, epoch_time = train_one_epoch(epoch)
        val_loss = validate()

        print("\n" + "-" * 70)
        print(f"Epoch      : {epoch}/{EPOCHS}")
        print(f"Train Loss : {train_loss:.6f}")
        print(f"Val Loss   : {val_loss:.6f}")
        print(f"Time       : {epoch_time:.2f} sec")
        print("-" * 70)

        # Let the Plateau Scheduler observe the validation loss
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
            }, BEST_MODEL_PATH)

            print(f"\nBest model saved -> {BEST_MODEL_PATH}")

    print("\n" + "=" * 70)
    print("Training Completed")
    print("=" * 70)
    print(f"Best Validation Loss : {best_loss:.6f}")
    print(f"Checkpoint           : {BEST_MODEL_PATH}")
    print("=" * 70)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train()
"""
==============================================================================
VHNet v2 (Multi-Task)
Loss Functions

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead : Babu Saheb

Loss Pipeline:
1. RGB Reconstruction: L1 + MS-SSIM + LPIPS + Sobel Edge
2. Semantic Segmentation: Cross-Entropy Loss
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import MS_SSIM

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# ============================================================
# L1 LOSS
# ============================================================

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)

# ============================================================
# MS-SSIM LOSS
# ============================================================

class MSSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = MS_SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, prediction, target):
        return 1.0 - self.metric(prediction, target)

# ============================================================
# SOBEL EDGE LOSS
# ============================================================

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient(self, image):
        channels = image.size(1)
        weight_x = self.sobel_x.repeat(channels, 1, 1, 1)
        weight_y = self.sobel_y.repeat(channels, 1, 1, 1)

        gx = F.conv2d(image, weight_x, padding=1, groups=channels)
        gy = F.conv2d(image, weight_y, padding=1, groups=channels)
        return gx, gy

    def forward(self, prediction, target):
        px, py = self.gradient(prediction)
        tx, ty = self.gradient(target)

        loss_x = F.l1_loss(px, tx)
        loss_y = F.l1_loss(py, ty)

        return loss_x + loss_y

# ============================================================
# LPIPS LOSS
# ============================================================

class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if LPIPS_AVAILABLE:
            self.metric = lpips.LPIPS(net="alex")
            self.enabled = True
        else:
            self.metric = None
            self.enabled = False

    def forward(self, prediction, target):
        if not self.enabled:
            return prediction.new_tensor(0.0)
        
        prediction = prediction * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.metric(prediction, target).mean()

# ============================================================
# SEGMENTATION LOSS (MODIFIED FOR STABILITY)
# ============================================================

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use reduction='mean' to ensure loss is averaged correctly
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, prediction_logits, target_mask):
        # Ensure target_mask is the right type
        target_mask = target_mask.long()
        
        # Stability check: Clamp logits to avoid extreme values before Softmax
        prediction_logits = torch.clamp(prediction_logits, min=-20, max=20)
        
        return self.loss_fn(prediction_logits, target_mask)

# ============================================================
# VHNET TOTAL MULTI-TASK LOSS
# ============================================================

class VHNetLoss(nn.Module):
    def __init__(
        self, 
        l1_weight=1.0, 
        msssim_weight=0.50, 
        lpips_weight=0.20, 
        edge_weight=0.10,
        seg_weight=1.0 # Added weight for Segmentation
    ):
        super().__init__()
        self.l1 = L1Loss()
        self.msssim = MSSSIMLoss()
        self.lpips = LPIPSLoss()
        self.edge = EdgeLoss()
        self.segmentation = SegmentationLoss()

        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.lpips_weight = lpips_weight
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight

    def forward(self, rgb_pred, rgb_gt, mask_pred, mask_gt):
        # ----------------------------------------------------
        # 1. RGB Reconstruction Losses
        # ----------------------------------------------------
        l1_loss = self.l1(rgb_pred, rgb_gt)
        msssim_loss = self.msssim(rgb_pred, rgb_gt)
        # Force LPIPS to run in FP32 to prevent Convolution Backward NaN explosions
        with torch.cuda.amp.autocast(enabled=False):
            lpips_loss = self.lpips(rgb_pred.float(), rgb_gt.float())
        edge_loss = self.edge(rgb_pred, rgb_gt)
        # ----------------------------------------------------
        # 2. Semantic Mask Loss
        # ----------------------------------------------------
        seg_loss = self.segmentation(mask_pred, mask_gt)

        # ----------------------------------------------------
        # TOTAL COMBINED LOSS
        # ----------------------------------------------------
        total_loss = (
            (self.l1_weight * l1_loss) +
            (self.msssim_weight * msssim_loss) +
            (self.lpips_weight * lpips_loss) +
            (self.edge_weight * edge_loss) +
            (self.seg_weight * seg_loss)
        )

        return {
            "loss": total_loss,
            "l1": l1_loss.detach(),
            "msssim": msssim_loss.detach(),
            "lpips": lpips_loss.detach(),
            "edge": edge_loss.detach(),
            "segmentation": seg_loss.detach(),
        }

# ============================================================
# CONFIGURATION INJECTION
# ============================================================

from config import (
    L1_WEIGHT,
    SSIM_WEIGHT,
    LPIPS_WEIGHT,
    EDGE_WEIGHT,
    SEGMENTATION_WEIGHT,
)

# ============================================================
# LOSS FACTORY
# ============================================================

def build_loss():
    """Build VHNet v2 Multi-Task loss function."""
    return VHNetLoss(
        l1_weight=L1_WEIGHT,
        msssim_weight=SSIM_WEIGHT,
        lpips_weight=LPIPS_WEIGHT,
        edge_weight=EDGE_WEIGHT,
        seg_weight=SEGMENTATION_WEIGHT,
    )

# ============================================================
# VERIFY LOSS
# ============================================================

@torch.no_grad()
def verify_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_loss().to(device)

    # Mocking config dimensions (e.g., 256 input, 512 output)
    B, C_rgb, H_hr, W_hr = 1, 3, 512, 512
    C_mask, H_lr, W_lr = 4, 256, 256

    rgb_pred = torch.rand(B, C_rgb, H_hr, W_hr, device=device)
    rgb_gt = torch.rand(B, C_rgb, H_hr, W_hr, device=device)

    # Mask logits (predictions) and long integers (ground truth targets)
    mask_pred = torch.randn(B, C_mask, H_lr, W_lr, device=device)
    mask_gt = torch.randint(0, C_mask, (B, H_lr, W_lr), device=device, dtype=torch.long)

    outputs = criterion(rgb_pred, rgb_gt, mask_pred, mask_gt)

    assert "loss" in outputs
    assert "segmentation" in outputs
    assert torch.is_tensor(outputs["loss"])

    return True

# ============================================================
# MAIN
# ============================================================

def main():
    if verify_loss():
        print("VHNet v2 (Multi-Task) Loss : Ready and Verified")

if __name__ == "__main__":
    main()
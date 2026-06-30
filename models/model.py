"""
==============================================================================
VHNet v2 (Multi-Task: Super-Resolution + Semantic Segmentation)

ISRO Hackathon 2026 - Problem Statement 10
Team : Void Hunter
Lead Developer : Babu Saheb

Description:
An end-to-end multi-task deep learning framework for satellite remote sensing.
Transforms low-visibility, monochrome Infrared (IR) imagery into high-fidelity, 
colorized RGB imagery (2x Super-Resolution) while simultaneously predicting a 
pixel-wise Semantic Segmentation mask for object interpretation to ensure 
semantic integrity.

Input   : Arbitrary Size Infrared (Dynamically padded for 16-divisibility)
Output 1: 2x High-Resolution Colorized RGB
Output 2: 1x Object Interpretation Mask
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. WEIGHT INITIALIZATION
# ============================================================

def initialize_weights(module):
    """Initializes weights using Kaiming Normal for convolutional layers."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# ============================================================
# 2. CORE BUILDING BLOCKS
# ============================================================

class ConvBNAct(nn.Module):
    """Standard Convolution -> Batch Normalization -> Activation (SiLU)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DoubleConv(nn.Module):
    """Two consecutive ConvBNAct blocks used in U-Net stages."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    """Residual connection to preserve gradient flow in deep networks."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels)
        self.conv2 = ConvBNAct(channels, channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity

class Up(nn.Module):
    """Bilinear upsampling followed by feature concatenation and convolution."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ============================================================
# 3. ATTENTION MECHANISMS (CBAM)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        mx = self.mlp(self.max(x))
        return x * self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg, mx], dim=1)
        att = self.conv(att)
        return x * self.sigmoid(att)

class CBAM(nn.Module):
    """Convolutional Block Attention Module to focus on spatial/channel features."""
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

# ============================================================
# 4. FEATURE EXTRACTION & REFINEMENT
# ============================================================

class MultiScaleBlock(nn.Module):
    """Inception-style block for extracting terrain features at multiple scales."""
    def __init__(self, channels):
        super().__init__()
        branch = channels // 4
        self.b1 = nn.Conv2d(channels, branch, 1, bias=False)
        self.b3 = nn.Conv2d(channels, branch, 3, padding=1, bias=False)
        self.b5 = nn.Conv2d(channels, branch, 5, padding=2, bias=False)
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, branch, 1, bias=False),
        )
        self.fuse = ConvBNAct(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.pool(x)], dim=1)
        return self.fuse(y)

class FeatureRefinement(nn.Module):
    """Combines Multi-Scale extraction, Residual connections, and Attention."""
    def __init__(self, channels):
        super().__init__()
        self.ms = MultiScaleBlock(channels)
        self.res = ResidualBlock(channels)
        self.att = CBAM(channels)

    def forward(self, x):
        identity = x
        x = self.ms(x)
        x = self.res(x)
        x = self.att(x)
        return x + identity

# ============================================================
# 5. U-NET STAGES
# ============================================================

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.refine = FeatureRefinement(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.refine(x)

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            DoubleConv(256, 512),
            FeatureRefinement(512),
            ResidualBlock(512),
        )

    def forward(self, x):
        return self.block(x)

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = Up(in_channels, skip_channels, out_channels)
        self.refine = FeatureRefinement(out_channels)

    def forward(self, x, skip):
        x = self.up(x, skip)
        return self.refine(x)

# ============================================================
# 6. MULTI-TASK OUTPUT HEADS
# ============================================================

class RGBHead(nn.Module):
    """Generates the base 1x resolution colorized image features."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(32, 32),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)

# Ensure your SuperResolutionHead in model.py looks like this for sharpness:
class SuperResolutionHead(nn.Module):
    """Upscales the generated features to 2x resolution with sharp details."""
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(3, 32), # Changed from 64 to 32 to match your checkpoint
            ConvBNAct(32, 32),
            nn.Conv2d(32, 32 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            ConvBNAct(32, 16),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.block(x)

class SegmentationHead(nn.Module):
    """Generates a semantic mask for object interpretation."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, 32),
            nn.Conv2d(32, num_classes, kernel_size=1) 
        )

    def forward(self, x):
        return self.block(x)

# ============================================================
# 7. MAIN VHNET V2 ARCHITECTURE
# ============================================================

class VHNet_v2(nn.Module):
    def __init__(self, num_classes=4):
        """
        Args:
            num_classes (int): Number of semantic object classes to detect.
                               Default is 4 (e.g., Background, Water, Forest, Man-made).
        """
        super().__init__()
        
        # --- Encoder ---
        self.stem = EncoderStage(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = EncoderStage(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = EncoderStage(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = EncoderStage(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        # --- Bottleneck ---
        self.bottleneck = Bottleneck()
        
        # --- Decoder ---
        self.decoder4 = DecoderStage(512, 256, 256)
        self.decoder3 = DecoderStage(256, 128, 128)
        self.decoder2 = DecoderStage(128, 64, 64)
        self.decoder1 = DecoderStage(64, 32, 32)
        
        # --- Multi-Task Heads ---
        self.rgb = RGBHead()
        self.super_resolution = SuperResolutionHead()
        self.segmentation = SegmentationHead(in_channels=32, num_classes=num_classes)

        # Apply weight initialization
        self.apply(initialize_weights)

    def forward(self, x):
        # 1. DYNAMIC PADDING (Solves the "Any Size" requirement)
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 2. ENCODER PASS
        s1 = self.stem(x)
        x_down = self.pool1(s1)
        
        s2 = self.encoder2(x_down)
        x_down = self.pool2(s2)
        
        s3 = self.encoder3(x_down)
        x_down = self.pool3(s3)
        
        s4 = self.encoder4(x_down)
        x_down = self.pool4(s4)
        
        # 3. BOTTLENECK PASS
        x_neck = self.bottleneck(x_down)
        
        # 4. DECODER PASS
        x_up = self.decoder4(x_neck, s4)
        x_up = self.decoder3(x_up, s3)
        x_up = self.decoder2(x_up, s2)
        x_base = self.decoder1(x_up, s1)
        
        # 5. MULTI-TASK PREDICTIONS
        rgb_base = self.rgb(x_base)
        
        # Use your trained SuperResolutionHead instead of bilinear bypass
        rgb_hr = self.super_resolution(rgb_base) 
        
        seg_mask = self.segmentation(x_base)

        # 6. DYNAMIC CROPPING (Restores exact requested dimensions)
        rgb_hr = rgb_hr[:, :, :h * 2, :w * 2]
        seg_mask = seg_mask[:, :, :h, :w]

        return rgb_hr, seg_mask

# ============================================================
# 8. MODEL SUMMARY TEST SCRIPT
# ============================================================

def model_summary():
    """Validates the network architecture against Hackathon specifications."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VHNet_v2(num_classes=4).to(device)
    
    # Simulating a dynamic, non-standard satellite tile size (e.g., 300x300)
    test_h, test_w = 300, 300
    x_test = torch.randn(1, 1, test_h, test_w).to(device)
    
    with torch.no_grad():
        rgb_out, seg_out = model(x_test)
        
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 65)
    print("VHNet v2 (Dynamic Multi-Task Network)")
    print("=" * 65)
    print(f"Total Parameters   : {params:,}")
    print(f"Trainable Params   : {trainable:,}")
    print("-" * 65)
    print("DYNAMIC INFERENCE TEST")
    print(f"Input IR Shape     : {list(x_test.shape)} (Grayscale, 1x)")
    print(f"Output RGB Shape   : {list(rgb_out.shape)} (Colorized, 2x Super Res)")
    print(f"Output Mask Shape  : {list(seg_out.shape)} (Segmentation, 1x)")
    print("=" * 65)

if __name__ == "__main__":
    model_summary()
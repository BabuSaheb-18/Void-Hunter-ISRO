import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Double Convolution Block
# ==========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Down Block
# ==========================================================

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Up Block
# ==========================================================

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2
            ]
        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


# ==========================================================
# Output Layer
# ==========================================================

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================================
# Lightweight U-Net
# ==========================================================

class IR2RGBUNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Lightweight)

        self.inc = DoubleConv(1, 32)

        self.down1 = Down(32, 64)

        self.down2 = Down(64, 128)

        self.down3 = Down(128, 256)

        # Bottleneck

        self.bridge = DoubleConv(256, 512)

        # Decoder

        self.up1 = Up(512, 256)

        self.up2 = Up(256, 128)

        self.up3 = Up(128, 64)

        self.up4 = Up(64, 32)

        # Output

        self.outc = OutConv(32, 3)

        self.activation = nn.Sigmoid()

    def forward(self, x):

        # Encoder

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        # Bridge

        x5 = self.bridge(x4)

        # Decoder

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outc(x)

        return self.activation(x)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    model = IR2RGBUNet()

    print(model)

    dummy = torch.randn(1, 1, 256, 256)

    output = model(dummy)

    print("\nInput Shape :", dummy.shape)

    print("Output Shape:", output.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal Parameters     : {total_params:,}")

    print(f"Trainable Parameters : {trainable_params:,}")
"""
Implements different version of CNN models.
"""
import torch
from torch import nn

class DoubleConvxReLU_NoNorm(nn.Module):
    """
    U-Net uses (3x3 convolution followed by ReLU activation) x 2.
    This class implements said code block.
    No normalization.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convxrelu2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.convxrelu2(x)

class DoubleConvxReLU_BatchNorm(nn.Module):
    """
    U-Net uses (3x3 convolution followed by ReLU activation) x 2.
    This class implements said code block.
    + BatchNorm
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convxrelu2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.convxrelu2(x)

class DeconvolutionUNet_Baseline(nn.Module):
    """
    U-Net-based network for image deconvolution - baseline model.
    Uses direct image as target.
    No normalization!
    """
    def __init__(self):
        super().__init__()

        self.enc0 = DoubleConvxReLU_NoNorm(in_channels=1, out_channels=16)
        self.enc1 = DoubleConvxReLU_NoNorm(in_channels=16, out_channels=32)
        self.enc2 = DoubleConvxReLU_NoNorm(in_channels=32, out_channels=64)
        self.enc3 = DoubleConvxReLU_NoNorm(in_channels=64, out_channels=128)
        self.enc4 = DoubleConvxReLU_NoNorm(in_channels=128, out_channels=256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConvxReLU_NoNorm(in_channels=256, out_channels=512)

        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # we have 256 from upsample and 256 from skip connection from enc4 to dec4
        # =>
        self.dec4 = DoubleConvxReLU_NoNorm(in_channels=512, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvxReLU_NoNorm(in_channels=256, out_channels=128)

        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = DoubleConvxReLU_NoNorm(in_channels=128, out_channels=64)

        self.up1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec1 = DoubleConvxReLU_NoNorm(in_channels=64, out_channels=32)

        self.up0 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec0 = DoubleConvxReLU_NoNorm(in_channels=32, out_channels=16)

        self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass
        :param x: [B, C=1, H=256, W=256]
        :return:
        """
        blurred_input = x

        e0 = self.enc0(x)

        e1 = self.enc1(self.pool(e0))

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        out = self.out(d0)

        return torch.clamp(out, 0, 1)

class DeconvolutionUNet_BatchNorm(nn.Module):
    """
    U-Net-based network for image deconvolution - baseline model.
    Uses direct image as target.
    Uses BatchNorm for normalization.
    """
    def __init__(self):
        super().__init__()

        self.enc0 = DoubleConvxReLU_BatchNorm(in_channels=1, out_channels=16)
        self.enc1 = DoubleConvxReLU_BatchNorm(in_channels=16, out_channels=32)
        self.enc2 = DoubleConvxReLU_BatchNorm(in_channels=32, out_channels=64)
        self.enc3 = DoubleConvxReLU_BatchNorm(in_channels=64, out_channels=128)
        self.enc4 = DoubleConvxReLU_BatchNorm(in_channels=128, out_channels=256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConvxReLU_BatchNorm(in_channels=256, out_channels=512)

        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec4 = DoubleConvxReLU_BatchNorm(in_channels=512, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvxReLU_BatchNorm(in_channels=256, out_channels=128)

        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = DoubleConvxReLU_BatchNorm(in_channels=128, out_channels=64)

        self.up1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec1 = DoubleConvxReLU_BatchNorm(in_channels=64, out_channels=32)

        self.up0 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec0 = DoubleConvxReLU_BatchNorm(in_channels=32, out_channels=16)

        self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass
        :param x: [B, C=1, H=256, W=256]
        :return:
        """
        blurred_input = x

        e0 = self.enc0(x)

        e1 = self.enc1(self.pool(e0))

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        out = self.out(d0)

        return torch.clamp(out, 0, 1)

class DeconvolutionUNet_Residual(nn.Module):
    """
    U-Net-based network for image deconvolution.
    Uses residual image as target.
    """
    def __init__(self):
        super().__init__()

        self.enc0 = DoubleConvxReLU_BatchNorm(in_channels=1, out_channels=16)
        self.enc1 = DoubleConvxReLU_BatchNorm(in_channels=16, out_channels=32)
        self.enc2 = DoubleConvxReLU_BatchNorm(in_channels=32, out_channels=64)
        self.enc3 = DoubleConvxReLU_BatchNorm(in_channels=64, out_channels=128)
        self.enc4 = DoubleConvxReLU_BatchNorm(in_channels=128, out_channels=256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConvxReLU_BatchNorm(in_channels=256, out_channels=512)

        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        # we have 256 from upsample and 256 from skip connection from enc4 to dec4
        # =>
        self.dec4 = DoubleConvxReLU_BatchNorm(in_channels=512, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvxReLU_BatchNorm(in_channels=256, out_channels=128)

        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = DoubleConvxReLU_BatchNorm(in_channels=128, out_channels=64)

        self.up1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.dec1 = DoubleConvxReLU_BatchNorm(in_channels=64, out_channels=32)

        self.up0 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.dec0 = DoubleConvxReLU_BatchNorm(in_channels=32, out_channels=16)

        self.out = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass
        :param x: [B, C=1, H=256, W=256]
        :return:
        """
        blurred_input = x

        e0 = self.enc0(x)

        e1 = self.enc1(self.pool(e0))

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e0], dim=1)
        d0 = self.dec0(d0)

        residual = self.out(d0)
        out = blurred_input + residual

        return torch.clamp(out, 0, 1)

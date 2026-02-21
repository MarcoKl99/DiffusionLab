import torch
import torch.nn as nn

from .shared import DownBlock, UpBlock
from .base_backbone import BaseBackbone


class CNNDenoiserLarge(BaseBackbone):
    """
    CNNDenoiser implementation (U-Net style) with larger architecture.
    """

    def __init__(
            self,
            in_channels: int = 1,
            base_channels: int = 128,
            time_emb_dim: int = 128
    ):
        super().__init__()

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (down-sampling path)
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 8, time_emb_dim)

        # Decoder (up-sampling path)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block1 = UpBlock(base_channels * 8, base_channels * 4, time_emb_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block3 = UpBlock(base_channels * 2, base_channels, time_emb_dim)

        # Output conv
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x0 = self.init_conv(x)

        # Encoder
        x1 = self.down1(x0, cond_emb)
        x1_pooled = self.pool1(x1)

        x2 = self.down2(x1_pooled, cond_emb)
        x2_pooled = self.pool2(x2)

        x3 = self.down3(x2_pooled, cond_emb)
        x3_pooled = self.pool3(x3)

        # Bottleneck
        bottleneck = self.bottleneck(x3_pooled, cond_emb)

        # Decoder with skip connections
        up1 = self.up1(bottleneck)
        up1 = self.up_block1(up1, x3, cond_emb)

        up2 = self.up2(up1)
        up2 = self.up_block2(up2, x2, cond_emb)

        up3 = self.up3(up2)
        up3 = self.up_block3(up3, x1, cond_emb)

        # Output
        out = self.out_conv(up3)

        return out

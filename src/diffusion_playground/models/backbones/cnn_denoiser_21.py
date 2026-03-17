import torch
import torch.nn as nn

from .shared import AttentionBlock, DownBlock, UpBlock
from .base_backbone import BaseBackbone


class CNNDenoiser21(BaseBackbone):
    """
    U-Net denoiser.
    Implemented using 3 blocks/stage × 3 stages = 9 encoder + 3 bottleneck + 9 decoder = 21 Conv blocks
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 128,
            time_emb_dim: int = 128,
    ):
        super().__init__()

        C = base_channels

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, C, kernel_size=3, padding=1)

        # Encoder
        # Stage 1
        self.down1a = DownBlock(C, C * 2, time_emb_dim)
        self.down1b = DownBlock(C * 2, C * 2, time_emb_dim)
        self.down1c = DownBlock(C * 2, C * 2, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2
        self.down2a = DownBlock(C * 2, C * 4, time_emb_dim)
        self.down2b = DownBlock(C * 4, C * 4, time_emb_dim)
        self.down2c = DownBlock(C * 4, C * 4, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3
        self.down3a = DownBlock(C * 4, C * 8, time_emb_dim)
        self.down3b = DownBlock(C * 8, C * 8, time_emb_dim)
        self.down3c = DownBlock(C * 8, C * 8, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck1 = DownBlock(C * 8, C * 8, time_emb_dim)
        self.attn_bot1 = AttentionBlock(C * 8)
        self.bottleneck2 = DownBlock(C * 8, C * 8, time_emb_dim)
        self.attn_bot2 = AttentionBlock(C * 8)
        self.bottleneck3 = DownBlock(C * 8, C * 8, time_emb_dim)

        # Decoder
        # Stage 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block3c = UpBlock(C * 8, C * 8, time_emb_dim)
        self.up_block3b = UpBlock(C * 8, C * 8, time_emb_dim)
        self.attn_8x8 = AttentionBlock(C * 8)
        self.up_block3a = UpBlock(C * 8, C * 4, time_emb_dim)

        # Stage 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block2c = UpBlock(C * 4, C * 4, time_emb_dim)
        self.up_block2b = UpBlock(C * 4, C * 4, time_emb_dim)
        self.attn_16x16 = AttentionBlock(C * 4)
        self.up_block2a = UpBlock(C * 4, C * 2, time_emb_dim)

        # Stage 1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block1c = UpBlock(C * 2, C * 2, time_emb_dim)
        self.up_block1b = UpBlock(C * 2, C * 2, time_emb_dim)
        self.up_block1a = UpBlock(C * 2, C, time_emb_dim)

        # Output projection back to image channels
        self.out_conv = nn.Conv2d(C, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x0 = self.init_conv(x)

        # Encoder
        s1a = self.down1a(x0, cond_emb)
        s1b = self.down1b(s1a, cond_emb)
        s1c = self.down1c(s1b, cond_emb)

        s2a = self.down2a(self.pool1(s1c), cond_emb)
        s2b = self.down2b(s2a, cond_emb)
        s2c = self.down2c(s2b, cond_emb)

        s3a = self.down3a(self.pool2(s2c), cond_emb)
        s3b = self.down3b(s3a, cond_emb)
        s3c = self.down3c(s3b, cond_emb)

        # Bottleneck (block -> attn -> block -> attn -> block)
        b = self.bottleneck1(self.pool3(s3c), cond_emb)
        b = self.attn_bot1(b)
        b = self.bottleneck2(b, cond_emb)
        b = self.attn_bot2(b)
        b = self.bottleneck3(b, cond_emb)

        # Decoder (each UpBlock consumes one skip, in reverse order)
        u = self.up_block3c(self.up3(b), s3c, cond_emb)
        u = self.up_block3b(u, s3b, cond_emb)
        u = self.attn_8x8(u)
        u = self.up_block3a(u, s3a, cond_emb)

        u = self.up_block2c(self.up2(u), s2c, cond_emb)
        u = self.up_block2b(u, s2b, cond_emb)
        u = self.attn_16x16(u)
        u = self.up_block2a(u, s2a, cond_emb)

        u = self.up_block1c(self.up1(u), s1c, cond_emb)
        u = self.up_block1b(u, s1b, cond_emb)
        u = self.up_block1a(u, s1a, cond_emb)

        return self.out_conv(u)

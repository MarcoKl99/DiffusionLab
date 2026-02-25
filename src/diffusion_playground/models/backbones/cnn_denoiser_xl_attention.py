import torch
import torch.nn as nn

from .shared import AttentionBlock, DownBlock, UpBlock
from .base_backbone import BaseBackbone


class CNNDenoiserXLAttention(BaseBackbone):
    """
    U-Net backbone designed for 64×64 images (Tiny ImageNet / full ImageNet).

    Extends CNNDenoiserLargeAttention with two key improvements:

    1. Four encoder/decoder stages (vs. three) so the bottleneck lands at 4×4,
       identical to what CNNDenoiserLargeAttention achieves on 32×32 CIFAR images.
       This preserves the same receptive field and attention cost at the bottleneck.

    Spatial resolution per stage (64×64 input):
        Encoder : 64 → 32 → 16 → 8 → 4  (bottleneck)
        Decoder :  4 →  8 → 16 → 32 → 64

    2. DDPM-style double bottleneck (block → attention → block).
       The original DDPM U-Net wraps the bottleneck attention between two residual
       blocks. At 4×4 this costs almost nothing but significantly increases the
       model's capacity to reason globally before the upsampling path begins.

    Attention is applied at three decoder resolutions:
        4×4  (16 tokens)  — bottleneck, global context, negligible cost
        8×8  (64 tokens)  — fine global context, very cheap
        16×16 (256 tokens) — medium-range context, still manageable
        32×32 is intentionally skipped (1 024 tokens → O(N²) cost becomes significant)

    Channel progression caps at base_channels * 8 to avoid memory explosion.
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

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down1 = DownBlock(C,     C * 2, time_emb_dim)   # 64×64
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DownBlock(C * 2, C * 4, time_emb_dim)   # 32×32
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DownBlock(C * 4, C * 8, time_emb_dim)   # 16×16
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DownBlock(C * 8, C * 8, time_emb_dim)   # 8×8
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck at 4×4 (DDPM-style: block → attention → block) ───────
        self.bottleneck1    = DownBlock(C * 8, C * 8, time_emb_dim)
        self.attn_bottleneck = AttentionBlock(C * 8)
        self.bottleneck2    = DownBlock(C * 8, C * 8, time_emb_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block1 = UpBlock(C * 8, C * 8, time_emb_dim)   # 4 → 8
        self.attn_8x8  = AttentionBlock(C * 8)

        self.up2        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block2  = UpBlock(C * 8, C * 4, time_emb_dim)  # 8 → 16
        self.attn_16x16 = AttentionBlock(C * 4)

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block3 = UpBlock(C * 4, C * 2, time_emb_dim)   # 16 → 32

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block4 = UpBlock(C * 2, C,     time_emb_dim)   # 32 → 64

        # Output projection back to image channels
        self.out_conv = nn.Conv2d(C, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x0 = self.init_conv(x)                         # (B, C,   64, 64)

        # Encoder
        x1 = self.down1(x0, cond_emb)                  # (B, C*2, 64, 64)
        x2 = self.down2(self.pool1(x1), cond_emb)      # (B, C*4, 32, 32)
        x3 = self.down3(self.pool2(x2), cond_emb)      # (B, C*8, 16, 16)
        x4 = self.down4(self.pool3(x3), cond_emb)      # (B, C*8,  8,  8)

        # Bottleneck (DDPM-style: block → attention → block)
        b = self.bottleneck1(self.pool4(x4), cond_emb) # (B, C*8,  4,  4)
        b = self.attn_bottleneck(b)
        b = self.bottleneck2(b, cond_emb)              # (B, C*8,  4,  4)

        # Decoder + skip connections
        u1 = self.up_block1(self.up1(b),  x4, cond_emb)  # (B, C*8,  8,  8)
        u1 = self.attn_8x8(u1)

        u2 = self.up_block2(self.up2(u1), x3, cond_emb)  # (B, C*4, 16, 16)
        u2 = self.attn_16x16(u2)

        u3 = self.up_block3(self.up3(u2), x2, cond_emb)  # (B, C*2, 32, 32)
        u4 = self.up_block4(self.up4(u3), x1, cond_emb)  # (B, C,   64, 64)

        return self.out_conv(u4)

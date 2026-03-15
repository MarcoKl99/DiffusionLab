import torch
import torch.nn as nn

from .shared import AttentionBlock, DownBlock, UpBlock
from .base_backbone import BaseBackbone


class CNNDenoiser15(BaseBackbone):
    """
    Deep U-Net denoiser for 32×32 images (CIFAR-10), inspired by the original DDPM architecture.

    The key difference from CNNDenoiserLargeAttention is using **2 residual blocks per
    resolution level** instead of one, which is the standard DDPM design. This roughly
    doubles the total depth without adding any extra pooling operations — preserving spatial
    resolution down to the same 4×4 bottleneck while significantly increasing capacity.

    Block count:
        Encoder  : 2 blocks × 3 stages =  6 DownBlocks
        Bottleneck: 3 blocks at 4×4    =  3 DownBlocks
        Decoder  : 2 blocks × 3 stages =  6 UpBlocks
                                  Total = 15 blocks (30 conv layers)

    Spatial resolution per stage (32×32 input):
        Encoder  : 32 → 16 → 8 → 4  (bottleneck)
        Decoder  :  4 →  8 → 16 → 32

    Attention is applied at:
        4×4   (16 tokens)  — DDPM-style: interleaved between bottleneck blocks
        8×8   (64 tokens)  — after first decoder block at this resolution
        16×16 (256 tokens) — after first decoder block at this resolution
        32×32 is intentionally skipped (1 024 tokens → O(N²) cost becomes significant)

    Channel progression: C → 2C → 4C → 8C (capped at base_channels * 8)
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
        # Stage 1: 32×32 — 2 blocks
        self.down1a = DownBlock(C,     C * 2, time_emb_dim)
        self.down1b = DownBlock(C * 2, C * 2, time_emb_dim)
        self.pool1  = nn.MaxPool2d(2)

        # Stage 2: 16×16 — 2 blocks
        self.down2a = DownBlock(C * 2, C * 4, time_emb_dim)
        self.down2b = DownBlock(C * 4, C * 4, time_emb_dim)
        self.pool2  = nn.MaxPool2d(2)

        # Stage 3: 8×8 — 2 blocks
        self.down3a = DownBlock(C * 4, C * 8, time_emb_dim)
        self.down3b = DownBlock(C * 8, C * 8, time_emb_dim)
        self.pool3  = nn.MaxPool2d(2)

        # ── Bottleneck at 4×4 (DDPM-style: block → attn → block → attn → block) ──
        self.bottleneck1  = DownBlock(C * 8, C * 8, time_emb_dim)
        self.attn_bot1    = AttentionBlock(C * 8)
        self.bottleneck2  = DownBlock(C * 8, C * 8, time_emb_dim)
        self.attn_bot2    = AttentionBlock(C * 8)
        self.bottleneck3  = DownBlock(C * 8, C * 8, time_emb_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Stage 3: 4 → 8 — 2 blocks, each consuming one skip from the encoder
        self.up3        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block3b = UpBlock(C * 8, C * 8, time_emb_dim)  # skip ← down3b
        self.attn_8x8   = AttentionBlock(C * 8)
        self.up_block3a = UpBlock(C * 8, C * 4, time_emb_dim)  # skip ← down3a

        # Stage 2: 8 → 16 — 2 blocks
        self.up2        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block2b = UpBlock(C * 4, C * 4, time_emb_dim)  # skip ← down2b
        self.attn_16x16 = AttentionBlock(C * 4)
        self.up_block2a = UpBlock(C * 4, C * 2, time_emb_dim)  # skip ← down2a

        # Stage 1: 16 → 32 — 2 blocks
        self.up1        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block1b = UpBlock(C * 2, C * 2, time_emb_dim)  # skip ← down1b
        self.up_block1a = UpBlock(C * 2, C,     time_emb_dim)  # skip ← down1a

        # Output projection back to image channels
        self.out_conv = nn.Conv2d(C, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # Initial projection
        x0 = self.init_conv(x)                              # (B, C,   32, 32)

        # ── Encoder ──────────────────────────────────────────────────────────
        s1a = self.down1a(x0,  cond_emb)                    # (B, C*2, 32, 32)
        s1b = self.down1b(s1a, cond_emb)                    # (B, C*2, 32, 32)

        s2a = self.down2a(self.pool1(s1b), cond_emb)        # (B, C*4, 16, 16)
        s2b = self.down2b(s2a,             cond_emb)        # (B, C*4, 16, 16)

        s3a = self.down3a(self.pool2(s2b), cond_emb)        # (B, C*8,  8,  8)
        s3b = self.down3b(s3a,             cond_emb)        # (B, C*8,  8,  8)

        # ── Bottleneck (block → attn → block → attn → block) ─────────────────
        b = self.bottleneck1(self.pool3(s3b), cond_emb)     # (B, C*8,  4,  4)
        b = self.attn_bot1(b)
        b = self.bottleneck2(b, cond_emb)                   # (B, C*8,  4,  4)
        b = self.attn_bot2(b)
        b = self.bottleneck3(b, cond_emb)                   # (B, C*8,  4,  4)

        # ── Decoder (each UpBlock consumes one skip, LIFO order) ─────────────
        u = self.up_block3b(self.up3(b), s3b, cond_emb)    # (B, C*8,  8,  8)
        u = self.attn_8x8(u)
        u = self.up_block3a(u,           s3a, cond_emb)    # (B, C*4,  8,  8)

        u = self.up_block2b(self.up2(u), s2b, cond_emb)    # (B, C*4, 16, 16)
        u = self.attn_16x16(u)
        u = self.up_block2a(u,           s2a, cond_emb)    # (B, C*2, 16, 16)

        u = self.up_block1b(self.up1(u), s1b, cond_emb)    # (B, C*2, 32, 32)
        u = self.up_block1a(u,           s1a, cond_emb)    # (B, C,   32, 32)

        return self.out_conv(u)

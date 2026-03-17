import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Self-attention block.
    This block applies multi head self-attention to all HxW elements of the feature maps and therefore
    provides a global context of attention across each entire feature map.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        """
        :param channels: Number of feature map channels (must be divisible by num_heads)
        :param num_heads: Number of attention heads
        """

        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Feature map of shape (B, C, H, W)
        :return: Feature map of same shape with self-attention applied
        """

        B, C, H, W = x.shape

        # Normalize and flatten
        h = self.norm(x).flatten(2).transpose(1, 2)

        # Perform self-attention
        h, _ = self.attn(h, h, h)

        # Restore the original shape
        h = h.transpose(1, 2).view(B, C, H, W)

        return x + h  # Residual connection


class DownBlock(nn.Module):
    """
    Down-sampling block with two conv layers and a residual connection.
    Note that here we only implement the convolutional processing, pooling is being done afterwards by
    the denoiser-network.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()

        # 1x1 conv to match input channels to output channels for residual connection
        self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # First conv + group-norm
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + group-norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.res_proj(x)


class UpBlock(nn.Module):
    """
    Up-sampling block with conv layers, skip connection, and residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        # The input will be concatenated with skip connection, so double the input channels
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()

        # 1x1 conv to project concatenated input (in_channels * 2) to out_channels for residual connection
        self.res_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the up-sampling convolutional block including adding skip connection.

        :param x: Input feature map.
        :param skip: Input map from the corresponding down-sampling convolutional block (skip connection).
        :param t_emb: Embedded time step.
        :return: Tensor of convoluted feature maps + skip information added
        """

        # Concatenate with skip connection
        x_cat = torch.cat([x, skip], dim=1)

        # First conv + group-norm
        h = self.conv1(x_cat)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + group-norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.res_proj(x_cat)

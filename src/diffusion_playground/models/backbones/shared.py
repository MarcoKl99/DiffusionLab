import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """
    Self-attention block for spatial feature maps.

    Each spatial position (h, w) is treated as a token with C-dimensional features.
    Applies multi-head self-attention over all H*W tokens so every position can attend
    to every other position — giving the network global context that convolutions alone
    cannot provide. The result is added back via a residual connection.

    Intended for use at low spatial resolutions (e.g. 4×4 or 8×8) where the O(N²)
    attention cost is negligible and the benefit of global reasoning is highest.
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

        # Normalize, flatten spatial dims into a sequence, apply attention, restore shape
        h = self.norm(x).flatten(2).transpose(1, 2)   # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)        # (B, C, H, W)

        return x + h  # residual connection


class DownBlock(nn.Module):
    """
    Down-sampling block with two conv layers.
    Note that here we only implement the convolutional processing, pooling is being done afterwards by
    the denoiser-network.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the down-sampling convolutional block.

        :param x: Input feature map.
        :param t_emb: Embedded time step.
        :return: Tensor of the convoluted feature maps, as input to subsequent pooling
                 (could be included in the down-block architecture as well ;) )
        """

        # First conv + group-norm
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + group-norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.relu(h)

        return h


class UpBlock(nn.Module):
    """
    Up-sampling block with conv layers and skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        # The input will be concatenated with skip connection, so double the input channels
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU()

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
        x = torch.cat([x, skip], dim=1)

        # First conv + group-norm
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + group-norm
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.relu(h)

        return h

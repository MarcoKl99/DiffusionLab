import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding as used in the original Transformer paper.
    Embeds scalar time steps into a higher-dimensional vector space.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time steps of shape (batch_size,) or (batch_size, 1)

        Returns:
            Time embeddings of shape (batch_size, embedding_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2

        # Create sinusoidal embeddings
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().view(-1, 1) * embeddings.view(1, -1)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class DownBlock(nn.Module):
    """
    Downsampling block with two conv layers.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, in_channels, H, W)
            t_emb: Time embeddings (batch_size, time_emb_dim)

        Returns:
            Output features (batch_size, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        return h


class UpBlock(nn.Module):
    """
    Upsampling block with two conv layers and skip connection.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        # The input will be concatenated with skip connection, so double the input channels
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, in_channels, H, W)
            skip: Skip connection from encoder (batch_size, in_channels, H, W)
            t_emb: Time embeddings (batch_size, time_emb_dim)

        Returns:
            Output features (batch_size, out_channels, H, W)
        """
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        return h

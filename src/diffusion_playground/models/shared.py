import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding.
    Embeds scalar time steps into a higher-dimensional vector space.
    See README.md in this package for more details.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the time embedding procedure.

        :param t: Time step to embed.
        :return: Embedded time step as tensor.
        """

        device = t.device
        half_dim = self.embedding_dim // 2

        # Create sinusoidal embeddings - use half dim for splitting across sine and cosine
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().view(-1, 1) * embeddings.view(1, -1)

        # Create the final embedding using sine and cosine
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
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

        # First conv + batch-norm
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + batch-norm
        h = self.conv2(h)
        h = self.bn2(h)
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
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

        # First conv + batch-norm
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)
        t_emb_proj = t_emb_proj.view(-1, t_emb_proj.shape[1], 1, 1)
        h = h + t_emb_proj

        # Second conv + batch-norm
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)

        return h

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


class ClassIndexTimeEmbedding(nn.Module):
    """
    Embedding of a class index to condition the model to a specific class, e.g. for the CIFAR-10 dataset.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create the conditioned embedding.

        :param t: Time-step to embed
        :param y: Class index to embed
        :return: Embedding
        """

        t_emb = self.time_embedding(t)
        class_emb = self.embedding(y)

        return t_emb + class_emb

import torch
import torch.nn as nn

from .cond_embedding import SinusoidalTimeEmbedding
from .backbones.base_backbone import BaseBackbone


class TimeConditionedModel(nn.Module):
    def __init__(
            self,
            backbone_model: BaseBackbone,
            time_emb_dim: int = 128,
    ):
        super().__init__()

        self.backbone_model = backbone_model
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure t is the right shape
        if t.dim() == 2:
            t = t.squeeze(1)

        # Get time embeddings
        cond_emb = self.time_embedding(t)

        return self.backbone_model(x, cond_emb)

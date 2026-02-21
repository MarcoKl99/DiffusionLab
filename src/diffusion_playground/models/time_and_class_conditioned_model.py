import torch
import torch.nn as nn

from .cond_embedding import ClassIndexTimeEmbedding
from .backbones.base_backbone import BaseBackbone


class TimeAndClassConditionedModel(nn.Module):
    def __init__(
            self,
            backbone_model: BaseBackbone,
            num_classes: int,
            time_emb_dim: int = 128,
    ):
        super().__init__()

        self.backbone_model = backbone_model
        self.cond_emb = ClassIndexTimeEmbedding(num_classes, time_emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get time embeddings
        cond_emb = self.cond_emb(t, y)

        return self.backbone_model(x, cond_emb)

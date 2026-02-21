from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        pass

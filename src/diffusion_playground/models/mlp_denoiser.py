import torch
import torch.nn as nn


class MLPDenoiser(nn.Module):
    """
    Simple Multi Layer Perceptron de-noising model.
    Used for Proof of Concept and testing of the training and inference pipeline.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # Add +1 for the timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Input din as output dim to get a new image
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t = t.float()
        x_0 = torch.cat([x, t], dim=1)

        return self.net(x_0)

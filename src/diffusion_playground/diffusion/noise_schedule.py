import torch


class LinearNoiseSchedule:
    def __init__(self, time_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.time_steps = time_steps
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

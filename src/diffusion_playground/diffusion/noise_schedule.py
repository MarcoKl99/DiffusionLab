import torch


class LinearNoiseSchedule:
    """
    Simple linear schedule for noising, computing the required values for alpha, beta, and the
    cumulative product alpha_bar for each time step, distributed over a given number of time steps.
    """

    def __init__(self, time_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.time_steps = time_steps
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

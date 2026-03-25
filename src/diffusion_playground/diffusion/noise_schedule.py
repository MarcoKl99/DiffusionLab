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

    def to(self, device: torch.device | str):
        """
        Move all schedule tensors to the specified device.

        :param device: Device to move tensors to (e.g., 'cuda', 'cpu')
        :return: Self for chaining
        """

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


class CosineNoiseSchedule:
    """
    Cosine noise schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
    (Nichol & Dhariwal, 2021). Produces a smoother noise schedule than the linear one,
    which helps avoid over-noising images at the end of the forward process and typically
    leads to better sample quality (lower FID).

    alpha_bar(t) = f(t) / f(0),  f(t) = cos((t/T + s) / (1 + s) * pi/2)^2

    Betas are derived from consecutive alpha_bar values and clipped to 0.999 to
    prevent numerical issues near t = T.
    """

    def __init__(self, time_steps: int, s: float = 0.008):
        self.time_steps = time_steps

        # Compute f(t) for t in {0, 1, ..., T}
        t = torch.arange(time_steps + 1, dtype=torch.float64)
        f = torch.cos((t / time_steps + s) / (1 + s) * torch.pi / 2) ** 2

        # alpha_bar(t) = f(t) / f(0), shape: [T+1]
        alpha_bars_full = f / f[0]

        # beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1), clipped for stability
        self.betas = (1 - alpha_bars_full[1:] / alpha_bars_full[:-1]).clamp(max=0.999).float()
        self.alphas = 1 - self.betas
        self.alpha_bars = alpha_bars_full[1:].float()

    def to(self, device: torch.device | str):
        """
        Move all schedule tensors to the specified device.

        :param device: Device to move tensors to (e.g., 'cuda', 'cpu')
        :return: Self for chaining
        """

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

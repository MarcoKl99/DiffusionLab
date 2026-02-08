import torch


class ForwardDiffusion:
    """
    Class to compute noised data for showcasing the idea of diffusion models.
    Only used for demonstration purposes.
    """

    def __init__(self, time_steps: int = 50, beta: float = 0.02):
        self.time_steps = time_steps
        self.beta = beta

    def diffuse(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Create a noising trajectory for some not-noised (raw) input data x0.

        :param x0: Not-noised (raw) input data to create a noising trajectory for
        :return: Tensor of steps of the noising schedule, each element represents a noised datapoint
                 --> Shape: [self.time_steps, *x0.shape]
        """

        x = x0.clone()
        trajectory = [x.clone()]

        sqrt_one_minus_beta = torch.sqrt(torch.tensor(1 - self.beta, device=x.device, dtype=x.dtype))
        sqrt_beta = torch.sqrt(torch.tensor(self.beta, device=x.device, dtype=x.dtype))

        for t in range(self.time_steps - 1):
            # Create Gaussian noise
            epsilon = torch.randn_like(x)
            x = sqrt_one_minus_beta * x + sqrt_beta * epsilon
            trajectory.append(torch.Tensor(x))

        return torch.stack(trajectory)

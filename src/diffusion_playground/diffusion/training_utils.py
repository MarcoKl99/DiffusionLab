import torch
from .noise_schedule import LinearNoiseSchedule


def sample_xt(
        x0: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        t: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples noised data based on the not-noised inputs x0 (batch) and the given noise schedule.
    If a tensor of time steps t is given, then this is used, otherwise the time steps are randomly chosen.

    :param x0: Not-noised batch of input data, expected shape [batch_size, feature_size]
               Example: (16, 2) for batch size of 16 and the moons dataset, (128, 28, 28) for a
               batch size of 128 and the MNIST dataset.
    :param noise_schedule: Schedule to apply noise to the input data, defines alpha, alpha_bar, beta
                           over a defined number of time steps.
    :param t: Batch of time steps to sample from
    :return: Tensor of noised data xt, added noise, and the time steps corresponding to xt
    """

    batch_size = x0.shape[0]
    device = x0.device

    # Get a random time step if no time step was given
    if t is None:
        t = torch.randint(
            low=0,
            high=noise_schedule.time_steps,
            size=(batch_size,),
            device=device
        )

    # Get the corresponding alpha bar
    alpha_bar_t = noise_schedule.alpha_bars[t].unsqueeze(1)

    # Dynamic reshape for correct broadcasting
    alpha_bar_t = alpha_bar_t.view(batch_size, *([1] * (x0.dim() - 1)))

    # Create the random noise
    noise = torch.randn_like(x0)

    # Add the noise to x0
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    return xt, noise, t

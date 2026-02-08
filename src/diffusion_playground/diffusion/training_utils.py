import torch
from .noise_schedule import LinearNoiseSchedule


def sample_xt(
        x0: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        t: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

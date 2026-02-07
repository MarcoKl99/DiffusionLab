import torch
from .noise_schedule import LinearNoiseSchedule


def sample_xt(x0: torch.Tensor, noise_schedule: LinearNoiseSchedule) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x0.shape[0]
    device = x0.device

    # Get a random time step
    t = torch.randint(
        low=0,
        high=noise_schedule.time_steps,
        size=(batch_size,),
        device=device
    )

    # Get the corresponding alpha bar
    alpha_bar_t = noise_schedule.alphas[t].unsqueeze(1)

    # Create the random noise
    noise = torch.randn_like(x0)

    # Add the noise to x0
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    return xt, noise, t.float().unsqueeze(1)

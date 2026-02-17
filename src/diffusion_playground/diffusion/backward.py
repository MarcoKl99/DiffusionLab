import torch
import torch.nn as nn

from .noise_schedule import LinearNoiseSchedule


def generate_samples(
        model: nn.Module,
        noise_schedule: LinearNoiseSchedule,
        image_shape: tuple[int, int, int],
        num_samples: int,
        device: torch.device | str,
) -> list[torch.Tensor]:
    """
    Generate samples by performing reverse diffusion (denoising) from pure noise.

    This function starts from pure Gaussian noise and iteratively denoises it using
    the trained model to generate samples from the learned distribution.

    :param model: Trained denoiser model
    :param noise_schedule: Noise schedule used during training
    :param image_shape: Shape of generated images as (channels, height, width)
    :param num_samples: Number of samples to generate
    :param device: Device to run generation on ('cpu', 'cuda', or torch.device)
    :return: Generated samples as list of images (h, w, c)
    """
    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)

    # Move model and noise schedule to device
    model.to(device)
    noise_schedule.to(device)
    model.eval()

    xt = torch.randn(num_samples, *image_shape, device=device)

    # Reverse diffusion loop (stochastic sampling)
    with torch.no_grad():
        for t in reversed(range(1, noise_schedule.time_steps + 1)):
            # Create time tensor for all samples
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            # Predict the noise
            pred_noise = model(xt, t_tensor)

            # Get schedule parameters
            beta_t = noise_schedule.betas[t - 1]
            alpha_t = noise_schedule.alphas[t - 1]
            alpha_bar_t = noise_schedule.alpha_bars[t - 1]

            # Compute the mean of the reverse distribution
            mean = (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)

            # Add noise (STOCHASTIC sampling for diversity!)
            if t > 1:
                # Add Gaussian noise scaled by beta
                noise = torch.randn_like(xt)
                sigma_t = torch.sqrt(beta_t)
                x_prev = mean + sigma_t * noise
            else:
                # No noise at the final step
                x_prev = mean

            # Update xt
            xt = x_prev

            # Print progress every 100 steps
            if t % 100 == 0:
                print(f"  Step {noise_schedule.time_steps - t + 1}/{noise_schedule.time_steps} (t={t})")

    images = []
    for x in xt:
        sample = (torch.permute(x, (1, 2, 0)) + 1) / 2
        img = torch.clamp(sample, 0, 1)
        images.append(img)

    return images


def generate_samples_deterministic(
        model: nn.Module,
        noise_schedule: LinearNoiseSchedule,
        image_shape: tuple[int, int, int],
        num_samples: int,
        device: torch.device | str,
        initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate samples using deterministic DDIM-style sampling (no added noise during reverse process).

    This variant of the reverse diffusion process does not add noise at each step,
    resulting in deterministic generation for a given initial noise. This can be useful
    for reproducible results or interpolation tasks.

    :param model: Trained denoiser model
    :param noise_schedule: Noise schedule used during training
    :param image_shape: Shape of generated images as (channels, height, width)
    :param num_samples: Number of samples to generate
    :param device: Device to run generation on ('cpu', 'cuda', or torch.device)
    :param initial_noise: Optional initial noise tensor. If None, pure Gaussian noise is generated.
                         If provided, num_samples is ignored and inferred from the tensor shape.
    :return: Generated samples as tensor of shape (num_samples, *image_shape)
    """
    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)

    # Move model and noise schedule to device
    model.to(device)
    noise_schedule.to(device)
    model.eval()

    # Initialize from noise
    if initial_noise is not None:
        xt = initial_noise.to(device)
        actual_num_samples = xt.shape[0]
    else:
        xt = torch.randn(num_samples, *image_shape, device=device)
        actual_num_samples = num_samples

    # Deterministic reverse diffusion loop (no added noise)
    with torch.no_grad():
        for t in reversed(range(1, noise_schedule.time_steps + 1)):
            # Create time tensor for all samples
            t_tensor = torch.full((actual_num_samples,), t, device=device, dtype=torch.long)

            # Predict the noise
            pred_noise = model(xt, t_tensor)

            # Get schedule parameters
            beta_t = noise_schedule.betas[t - 1]
            alpha_t = noise_schedule.alphas[t - 1]
            alpha_bar_t = noise_schedule.alpha_bars[t - 1]

            # Compute the mean (deterministic update, no added noise)
            xt = (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)

    return xt

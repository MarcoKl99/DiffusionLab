import torch

from src.diffusion_playground.diffusion.noise_schedule import LinearNoiseSchedule


def test_linear_noise_schedule():
    # Create an instance of the linear noise schedule
    schedule = LinearNoiseSchedule(time_steps=10, beta_start=0.01, beta_end=0.1)

    for i in range(len(schedule.betas) - 1):
        # Test that alpha is 1 - beta
        beta = schedule.betas[i]
        alpha = schedule.alphas[i]
        assert torch.allclose(alpha, 1 - beta)

        # Test that alpha_bar is the cumulative product of all previous alpha values
        alpha_previous = schedule.alphas[:i + 1]
        cumulative_prod = torch.prod(alpha_previous)
        alpha_bar = schedule.alpha_bars[i]
        assert torch.allclose(alpha_bar, cumulative_prod)

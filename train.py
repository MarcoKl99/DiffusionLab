import torch

from src.diffusion_playground.data.toy_datasets import load_toy_dataset
from src.diffusion_playground.models.mlp_denoiser import MLPDenoiser
from src.diffusion_playground.diffusion.noise_schedule import LinearNoiseSchedule
from src.diffusion_playground.training.denoiser_trainer import train_denoiser


# Dataset
data = load_toy_dataset("moons", n_samples=1_000)
data = torch.Tensor(data)

# Model
model = MLPDenoiser()

# Schedule
schedule = LinearNoiseSchedule(time_steps=100)

# Training
train_denoiser(model, data, schedule, epochs=1_000)

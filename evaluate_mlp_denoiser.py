import torch
import torch.nn.functional as F
from torchinfo import summary

from src.diffusion_playground.models.mlp_denoiser import MLPDenoiser
from src.diffusion_playground.data.toy_datasets import load_toy_dataset
from src.diffusion_playground.diffusion.noise_schedule import LinearNoiseSchedule
from src.diffusion_playground.visualization.plot import show_denoising_steps_2d

### Setup ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
time_steps = 100
visualize_ever = 10

### Load Data and Model ###
data = torch.Tensor(load_toy_dataset("moons", n_samples=1_000)).to(device)
idx = torch.randint(0, data.shape[0], (batch_size,))
x0_eval = data[idx]

model = MLPDenoiser()
model.load_state_dict(torch.load('data/models/mlp_denoiser.pth'))
model.to(device)
model.eval()

schedule = LinearNoiseSchedule(time_steps=time_steps)

### Create pure Noise ###
xt = torch.randn_like(x0_eval).to(device)

### Track the Steps for Visualizations later
x_steps = []

### Reverse Loop ###
for t in reversed(range(1, time_steps + 1)):
    # The current time step for every datapoint (same for all datapoints)
    t_tensor = torch.full((batch_size, 1), t, device=device, dtype=torch.float32)

    # Predict the noise that was "added" at this time step - Reverse diffusion process
    with torch.no_grad():
        pred_noise = model(xt, t_tensor)

    # Re-calculate the previous datapoints (note that the weights alpha_hat and beta must be included!)
    beta_t = schedule.betas[t - 1]
    alpha_t = schedule.alphas[t - 1]
    alpha_bar_t = schedule.alpha_bars[t - 1]

    # Step back
    x_prev = (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)

    # Update xt
    xt = x_prev

    # Track steps for visualization
    if t % visualize_ever == 0 or t == 1:
        x_steps.append((xt.clone(), pred_noise.clone()))

# Visualize
for i, (xt, pred_noise) in enumerate(x_steps):
    save_path = f"experiment_results/mlp_linear_only_small/step_{i * visualize_ever}_of_{time_steps}.png"
    show_denoising_steps_2d(x0_eval, xt, pred_noise, f"Step {i * visualize_ever} / {time_steps}", save_path)

# Print out the MSE against the original
mse_final = F.mse_loss(xt, x0_eval)
print(f"Final MSE: {mse_final.item():.4f}")

summary(model)

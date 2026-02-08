import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..diffusion.training_utils import sample_xt
from ..diffusion.noise_schedule import LinearNoiseSchedule


def train_denoiser(
    model: nn.Module,
    data: torch.Tensor,
    noise_schedule: LinearNoiseSchedule,
    epochs: int = 1_000,
    lr: float = 1e-3,
    batch_size: int = 128,
):
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check the device available for training and send the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Training Loop
    for epoch in tqdm(range(epochs)):
        # Get a random batch (batch_size random indexes)
        idx = torch.randint(0, data.shape[0] - 1, (batch_size,))
        x0 = data[idx]

        # Sample a noised version of the batch
        xt, noise, t = sample_xt(x0, noise_schedule)

        # Make the prediction
        pred_noise = model(xt, t)

        # Calculate the loss
        loss = F.mse_loss(pred_noise, noise)

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss after every 10th of the training
        step_width_output = epochs // 10
        if epoch % step_width_output == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

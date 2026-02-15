import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from ..diffusion.training_utils import sample_xt
from ..diffusion.noise_schedule import LinearNoiseSchedule


def train_denoiser(
        model: nn.Module,
        data: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        epochs: int = 1_000,
        lr: float = 1e-3,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
        save_every: int = 1_000,
) -> None:
    """
    Train a model on the given data.
    The noised samples are generated within the training loop to learn to sample from the learned
    probability distribution after training.

    :param model: Model to train
    :param data: Tensor of not-noised data (raw) to train the model on - noised samples are created in the process
    :param noise_schedule: Schedule to create noised samples with
    :param epochs: Number of epochs to train
    :param lr: Learning rate
    :param batch_size: Batch size
    :param checkpoint_dir: Directory to save checkpoints (None = no saving)
    :param save_every: Save checkpoint every N epochs
    """

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check the device available for training and send the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Move noise schedule to device (it's tiny, no memory concerns)
    noise_schedule.to(device)

    # Ensure data stays on CPU (datasets are usually too large for GPU memory)
    # Only batches will be moved to GPU during training
    data = data.cpu()

    # Setup checkpoint directory
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_path}")

    best_loss = float('inf')

    # Training Loop
    for epoch in tqdm(range(epochs)):
        # Get a random batch (batch_size random indexes)
        # Create indices on CPU to match the data tensor device
        idx = torch.randint(0, data.shape[0], (batch_size,), device='cpu')
        x0 = data[idx].to(device)

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

        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'noise_schedule_params': {
                    'time_steps': noise_schedule.time_steps,
                    'beta_start': noise_schedule.betas[0].item(),
                    'beta_end': noise_schedule.betas[-1].item(),
                }
            }
            checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_file)

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_checkpoint_file = checkpoint_path / "best_model.pt"
                torch.save(checkpoint, best_checkpoint_file)
                print(f"\n✓ New best model saved at epoch {epoch + 1} with loss {loss.item():.6f}")


def load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu"
) -> dict:
    """
    Load a model checkpoint.

    :param model: Model to load weights into
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: Optional optimizer to load state into
    :param device: Device to load the model to
    :return: Dictionary with checkpoint metadata (epoch, loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'noise_schedule_params': checkpoint.get('noise_schedule_params', {})
    }

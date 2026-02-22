import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..diffusion.training_utils import sample_xt
from ..diffusion.noise_schedule import LinearNoiseSchedule
from ..models.time_conditioned_model import TimeConditionedModel
from ..models.time_and_class_conditioned_model import TimeAndClassConditionedModel
from .utils import setup_training, setup_checkpoint_resume, save_epoch_checkpoint


def train_denoiser(
        model: TimeConditionedModel,
        data: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        epochs: int = 1_000,
        lr: float = 1e-3,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
        save_every: int = 1_000,
        resume: bool = True,
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
    :param resume: If True, automatically resume from the latest checkpoint if available
    """
    optimizer, device, data = setup_training(model, data, noise_schedule, lr)
    start_epoch, best_loss, checkpoint_path = setup_checkpoint_resume(
        checkpoint_dir, resume, model, optimizer, device
    )

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1} / {epochs}...")

        # Shuffle dataset at the start of each epoch
        perm = torch.randperm(data.shape[0])

        epoch_loss = 0.0
        num_batches = 0

        # Iterate over the full dataset in batches
        for i in tqdm(range(0, data.shape[0], batch_size), leave=False):
            idx = perm[i:i + batch_size]
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

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        print(f"  loss: {epoch_loss:.6f}")

        # Save checkpoint
        if checkpoint_path is not None and (epoch + 1) % save_every == 0:
            best_loss = save_epoch_checkpoint(
                checkpoint_path, epoch, model, optimizer, epoch_loss, noise_schedule, best_loss
            )


def train_conditioned_denoiser(
        model: TimeAndClassConditionedModel,
        data: torch.Tensor,
        labels: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        epochs: int = 1_000,
        lr: float = 1e-3,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
        save_every: int = 1_000,
        resume: bool = True,
) -> None:
    """
    Train a class-conditioned model on the given data.
    The noised samples are generated within the training loop to learn to sample from the learned
    probability distribution after training.

    :param model: Model to train
    :param data: Tensor of not-noised data (raw) to train the model on - noised samples are created in the process
    :param labels: Tensor of class labels corresponding to each data sample
    :param noise_schedule: Schedule to create noised samples with
    :param epochs: Number of epochs to train
    :param lr: Learning rate
    :param batch_size: Batch size
    :param checkpoint_dir: Directory to save checkpoints (None = no saving)
    :param save_every: Save checkpoint every N epochs
    :param resume: If True, automatically resume from the latest checkpoint if available
    """
    optimizer, device, data = setup_training(model, data, noise_schedule, lr)
    start_epoch, best_loss, checkpoint_path = setup_checkpoint_resume(
        checkpoint_dir, resume, model, optimizer, device
    )

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch + 1} / {epochs}...")

        # Shuffle dataset at the start of each epoch
        perm = torch.randperm(data.shape[0])

        epoch_loss = 0.0
        num_batches = 0

        # Iterate over the full dataset in batches
        for i in tqdm(range(0, data.shape[0], batch_size), leave=False):
            idx = perm[i:i + batch_size]
            x0 = data[idx].to(device)
            y = labels[idx].to(device)

            # Sample a noised version of the batch
            xt, noise, t = sample_xt(x0, noise_schedule)

            # Make the prediction
            pred_noise = model(xt, t, y)

            # Calculate the loss
            loss = F.mse_loss(pred_noise, noise)

            # Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        print(f"  loss: {epoch_loss:.6f}")

        # Save checkpoint
        if checkpoint_path is not None and (epoch + 1) % save_every == 0:
            best_loss = save_epoch_checkpoint(
                checkpoint_path, epoch, model, optimizer, epoch_loss, noise_schedule, best_loss
            )


def load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
) -> dict:
    """
    Load a model checkpoint from local disk.

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

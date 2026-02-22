import re
import torch
from pathlib import Path

from ..diffusion.noise_schedule import LinearNoiseSchedule


def setup_training(
        model: torch.nn.Module,
        data: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        lr: float,
) -> tuple[torch.optim.Optimizer, torch.device, torch.Tensor]:
    """
    Create the optimizer, determine the training device, and move model and noise schedule to it.
    Data is kept on CPU — only batches are moved to the device during training.

    :param model: Model to train
    :param data: Full training dataset tensor
    :param noise_schedule: Noise schedule used during training
    :param lr: Learning rate for the Adam optimizer
    :return: (optimizer, device, data_on_cpu)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Move noise schedule to device (it's tiny, no memory concerns)
    noise_schedule.to(device)

    # Ensure data stays on CPU (datasets are usually too large for GPU memory)
    data = data.cpu()

    return optimizer, device, data


def setup_checkpoint_resume(
        checkpoint_dir: str | None,
        resume: bool,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> tuple[int, float, Path | None]:
    """
    Set up the checkpoint directory and optionally resume training from the latest checkpoint.

    :param checkpoint_dir: Directory to look for / save checkpoints (None = no checkpointing)
    :param resume: If True, resume from the latest checkpoint when one exists
    :param model: Model whose weights are restored on resume
    :param optimizer: Optimizer whose state is restored on resume
    :param device: Device to map checkpoint tensors to
    :return: (start_epoch, best_loss, checkpoint_path)
             start_epoch is 0 when starting fresh, checkpoint_path is None when checkpoint_dir is None
    """
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = None

    if checkpoint_dir is None:
        return start_epoch, best_loss, checkpoint_path

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_path}")

    if resume:
        checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))

        if checkpoint_files:
            # Find the latest checkpoint by epoch number
            def get_epoch_num(path):
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', path.name)
                return int(match.group(1)) if match else 0

            latest_checkpoint = max(checkpoint_files, key=get_epoch_num)

            print(f"\n{'=' * 60}")
            print(f"Resuming from checkpoint: {latest_checkpoint.name}")
            print(f"{'=' * 60}")

            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"✓ Previous loss: {checkpoint['loss']:.6f}\n")
        else:
            print("No existing checkpoints found. Starting from scratch.\n")

    return start_epoch, best_loss, checkpoint_path


def save_epoch_checkpoint(
        checkpoint_path: Path,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_loss: float,
        noise_schedule: LinearNoiseSchedule,
        best_loss: float,
) -> float:
    """
    Save a periodic checkpoint and overwrite the best-model file when loss improves.

    :param checkpoint_path: Directory to write checkpoint files into
    :param epoch: Current epoch index (0-based)
    :param model: Model to snapshot
    :param optimizer: Optimizer to snapshot
    :param epoch_loss: Average loss for this epoch
    :param noise_schedule: Noise schedule (its hyperparameters are stored in the checkpoint)
    :param best_loss: Best loss seen so far (used to decide whether to update best_model.pt)
    :return: Updated best_loss
    """

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'noise_schedule_params': {
            'time_steps': noise_schedule.time_steps,
            'beta_start': noise_schedule.betas[0].item(),
            'beta_end': noise_schedule.betas[-1].item(),
        }
    }
    checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
    torch.save(checkpoint, checkpoint_file)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_checkpoint_file = checkpoint_path / "best_model.pt"
        torch.save(checkpoint, best_checkpoint_file)
        print(f"\n✓ New best model saved at epoch {epoch + 1} with loss {epoch_loss:.6f}")

    return best_loss

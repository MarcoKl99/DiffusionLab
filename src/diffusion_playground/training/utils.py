import re
import torch
import torch.nn as nn
from pathlib import Path

from ..diffusion.noise_schedule import LinearNoiseSchedule


class EMA:
    """
    Exponential Moving Average of model parameters (DDPM paper, decay=0.9999).

    Maintains shadow weights updated after each optimizer step as:
        shadow = decay * shadow + (1 - decay) * param

    Use apply_shadow() / restore() to temporarily swap the EMA weights into
    the model for evaluation, then swap the training weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def update(self, model: nn.Module) -> None:
        """Call after every optimizer.step()."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """
        Copy EMA weights into the model for evaluation.
        Returns the original weights so they can be restored afterwards.
        """
        original: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            original[name] = param.data.clone()
            param.data.copy_(self.shadow[name])
        return original

    def restore(self, model: nn.Module, original: dict[str, torch.Tensor]) -> None:
        """Restore training weights after evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(original[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Move noise schedule to device (it's tiny, no memory concerns)
    noise_schedule.to(device)

    # Ensure data stays on CPU (datasets are usually too large for GPU memory)
    data = data.cpu()

    # Validate that data is in the expected [-1, 1] range
    data_min = data.min().item()
    data_max = data.max().item()
    if data_min >= 0.0 or data_min < -1.0 - 1e-3 or data_max > 1.0 + 1e-3:
        raise ValueError(f"Data is in range [{data_min:.3f}, {data_max:.3f}] - expexted to be in [-1, 1]")

    return optimizer, device, data


def setup_ema(
        model: nn.Module,
        decay: float,
        checkpoint_path: Path | None,
        device: torch.device,
) -> EMA:
    """
    Create an EMA object initialised from the current model weights.
    If a checkpoint directory is provided and the latest checkpoint contains
    a saved EMA state, that state is restored automatically.

    Call this *after* setup_checkpoint_resume so the model already holds the
    resumed weights before the EMA shadow is initialised.
    """
    ema = EMA(model, decay=decay)

    if checkpoint_path is not None:
        checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            def get_epoch_num(path):
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', path.name)
                return int(match.group(1)) if match else 0

            latest = max(checkpoint_files, key=get_epoch_num)
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            if 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'])
                print("✓ Restored EMA shadow weights from checkpoint")

    return ema


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
        ema: EMA | None = None,
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
    :param ema: Optional EMA object whose shadow weights are saved alongside the model weights
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
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    # Delete any existing epoch checkpoints before saving the new one.
    # Truncate to 0 bytes first so that even if the OS/Google Drive moves the
    # file to trash instead of permanently deleting it, no storage is wasted.
    for old_checkpoint in checkpoint_path.glob("checkpoint_epoch_*.pt"):
        old_checkpoint.write_bytes(b'')
        old_checkpoint.unlink()

    checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
    torch.save(checkpoint, checkpoint_file)

    return best_loss

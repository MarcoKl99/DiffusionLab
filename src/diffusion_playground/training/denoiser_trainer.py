import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import re
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from ..diffusion.training_utils import sample_xt
from ..diffusion.noise_schedule import LinearNoiseSchedule


def _upload_to_s3(local_path: Path, s3_bucket: str, s3_key: str) -> bool:
    """Upload a file to S3 bucket."""
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_path), s3_bucket, s3_key)
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"⚠️  Failed to upload to S3: {e}")
        return False


def _download_from_s3(s3_bucket: str, s3_key: str, local_path: Path) -> bool:
    """Download a file from S3 bucket."""
    try:
        s3_client = boto3.client('s3')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(s3_bucket, s3_key, str(local_path))
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"⚠️  Failed to download from S3: {e}")
        return False


def _list_s3_checkpoints(s3_bucket: str, s3_prefix: str) -> list[dict]:
    """List checkpoint files in S3 bucket with their metadata."""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_prefix
        )

        if 'Contents' not in response:
            return []

        checkpoints = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.pt') and 'checkpoint_epoch_' in key:
                checkpoints.append({
                    'key': key,
                    'last_modified': obj['LastModified'],
                    'size': obj['Size']
                })
        return checkpoints
    except (ClientError, NoCredentialsError) as e:
        print(f"⚠️  Failed to list S3 checkpoints: {e}")
        return []


def train_denoiser(
        model: nn.Module,
        data: torch.Tensor,
        noise_schedule: LinearNoiseSchedule,
        epochs: int = 1_000,
        lr: float = 1e-3,
        batch_size: int = 128,
        checkpoint_dir: str | None = None,
        save_every: int = 1_000,
        resume: bool = True,
        s3_bucket: str | None = None,
        s3_prefix: str = "checkpoints",
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
    :param s3_bucket: S3 bucket name to upload checkpoints (None = no S3 upload)
    :param s3_prefix: Prefix/path within S3 bucket for checkpoints
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
    start_epoch = 0
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_path}")

        if s3_bucket:
            print(f"Checkpoints will be uploaded to S3: s3://{s3_bucket}/{s3_prefix}")

        # Check for existing checkpoints to resume from
        if resume:
            checkpoint_files = []

            # If S3 is configured, prioritize S3 (ignore local checkpoints)
            if s3_bucket:
                print("Checking S3 for checkpoints...")
                s3_checkpoints = _list_s3_checkpoints(s3_bucket, s3_prefix)

                if s3_checkpoints:
                    # Find the latest checkpoint by epoch number from S3
                    def get_epoch_num_from_key(s3_obj):
                        match = re.search(r'checkpoint_epoch_(\d+)\.pt', s3_obj['key'])
                        return int(match.group(1)) if match else 0

                    latest_s3_checkpoint = max(s3_checkpoints, key=get_epoch_num_from_key)
                    latest_epoch = get_epoch_num_from_key(latest_s3_checkpoint)

                    # Download the latest checkpoint from S3
                    checkpoint_filename = f"checkpoint_epoch_{latest_epoch}.pt"
                    local_checkpoint_path = checkpoint_path / checkpoint_filename

                    print(f"📥 Downloading checkpoint from S3: {latest_s3_checkpoint['key']}")
                    if _download_from_s3(s3_bucket, latest_s3_checkpoint['key'], local_checkpoint_path):
                        checkpoint_files = [local_checkpoint_path]
                        print(f"✓ Downloaded checkpoint successfully")
            else:
                # Only check local checkpoints if S3 is not configured
                checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))

            if checkpoint_files:
                # Find the latest checkpoint by epoch number
                def get_epoch_num(path):
                    match = re.search(r'checkpoint_epoch_(\d+)\.pt', path.name)
                    return int(match.group(1)) if match else 0

                latest_checkpoint = max(checkpoint_files, key=get_epoch_num)
                latest_epoch = get_epoch_num(latest_checkpoint)

                print(f"\n{'=' * 60}")
                print(f"🔄 Resuming from checkpoint: {latest_checkpoint.name}")
                print(f"{'=' * 60}")

                # Load checkpoint
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']

                print(f"✓ Resumed from epoch {start_epoch}")
                print(f"✓ Previous loss: {checkpoint['loss']:.6f}\n")
            else:
                print("No existing checkpoints found. Starting from scratch.\n")

    best_loss = float('inf')

    # Training Loop - start from start_epoch
    for epoch in tqdm(range(start_epoch, epochs)):
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

            # Upload to S3 if configured
            if s3_bucket:
                s3_key = f"{s3_prefix}/checkpoint_epoch_{epoch + 1}.pt"
                if _upload_to_s3(checkpoint_file, s3_bucket, s3_key):
                    print(f"📤 Uploaded checkpoint to S3: s3://{s3_bucket}/{s3_key}")

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_checkpoint_file = checkpoint_path / "best_model.pt"
                torch.save(checkpoint, best_checkpoint_file)

                # Upload best model to S3 if configured
                if s3_bucket:
                    s3_best_key = f"{s3_prefix}/best_model.pt"
                    if _upload_to_s3(best_checkpoint_file, s3_bucket, s3_best_key):
                        print(f"\n✓ New best model saved at epoch {epoch + 1} with loss {loss.item():.6f}")
                        print(f"📤 Uploaded best model to S3: s3://{s3_bucket}/{s3_best_key}")
                else:
                    print(f"\n✓ New best model saved at epoch {epoch + 1} with loss {loss.item():.6f}")


def load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
        s3_bucket: str | None = None,
        s3_key: str | None = None,
) -> dict:
    """
    Load a model checkpoint from local disk or S3.

    :param model: Model to load weights into
    :param checkpoint_path: Path to the checkpoint file (local path, or download destination if loading from S3)
    :param optimizer: Optional optimizer to load state into
    :param device: Device to load the model to
    :param s3_bucket: Optional S3 bucket name to download checkpoint from
    :param s3_key: Optional S3 key for the checkpoint file
    :return: Dictionary with checkpoint metadata (epoch, loss, etc.)
    """
    checkpoint_file = Path(checkpoint_path)

    # Download from S3 if configured
    if s3_bucket and s3_key:
        print(f"📥 Downloading checkpoint from S3: s3://{s3_bucket}/{s3_key}")
        if _download_from_s3(s3_bucket, s3_key, checkpoint_file):
            print(f"✓ Downloaded checkpoint successfully")
        else:
            raise RuntimeError(f"Failed to download checkpoint from S3: s3://{s3_bucket}/{s3_key}")

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

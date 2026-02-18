from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..diffusion.noise_schedule import LinearNoiseSchedule
from ..diffusion.backward import generate_samples
from ..training.denoiser_trainer import load_checkpoint


def generate_samples_from_checkpoints(
        model: nn.Module,
        model_name: str,
        device: torch.device | str,
        checkpoint_epochs: list[int],
        checkpoint_dir: str | Path,
        output_dir: str | Path,
        noise_schedule: LinearNoiseSchedule,
        image_shape: tuple[int, int, int],
        grid_size: tuple[int, int] = (3, 3),
) -> None:
    """
    Generate images from multiple checkpoints and save visualizations.

    This function loads checkpoints at specified epochs, performs reverse diffusion
    to generate samples from noise, and saves the results as image grids.

    :param model: Model instance to load checkpoints into (should match the architecture used during training)
    :param model_name: Name of the model that is evaluated
    :param device: Device to run generation on ('cpu', 'cuda', or torch.device)
    :param checkpoint_epochs: List of epoch numbers to generate samples from
    :param checkpoint_dir: Directory containing the checkpoint files
    :param output_dir: Directory to save the generated visualizations
    :param noise_schedule: Noise schedule used during training
    :param image_shape: Shape of generated images as (channels, height, width)
    :param grid_size: Grid dimensions for visualization as (rows, cols)
    """
    # Convert paths to Path objects
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    # Determine the number of samples based on the grid size
    num_samples = grid_size[0] * grid_size[1]

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert device to torch.device if string
    if isinstance(device, str):
        device = torch.device(device)

    # Move model and noise schedule to device
    model.to(device)
    noise_schedule.to(device)

    print(f"Output directory: {output_path}")
    print(f"Checkpoints: {checkpoint_epochs}")

    for epoch in checkpoint_epochs:
        print(f"\n{'=' * 60}")
        print(f"Processing checkpoint: epoch {epoch}")
        print(f"{'=' * 60}")

        # Define checkpoint path
        cp_name = f"checkpoint_epoch_{epoch}.pt"
        checkpoint_file = checkpoint_path / cp_name

        # Check if checkpoint exists
        if not checkpoint_file.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_file}")
            print(f"   Skipping...")
            continue

        # Load checkpoint
        checkpoint_info = load_checkpoint(model, str(checkpoint_file), device=str(device))

        # Generate samples using reverse diffusion
        print(f"Generating {num_samples} samples...")
        generated_images = generate_samples(
            model=model,
            noise_schedule=noise_schedule,
            image_shape=image_shape,
            num_samples=num_samples,
            device=device,
        )
        print(f"✓ Generation complete")

        # Create visualization grid
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

        # Handle both single row/col and multidimensional grids
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        for idx, ax in enumerate(axes.flat):
            if idx < num_samples:
                # Handle different image formats
                if image_shape[0] == 1:
                    # Grayscale: squeeze channel dimension
                    ax.imshow(generated_images[idx, 0].cpu())
                else:
                    # RGB
                    ax.imshow(generated_images[idx].cpu())
            ax.axis("off")

        # Add informative title
        title = f"Model: {model_name} - {cp_name}\n"
        title += f"Epoch: {checkpoint_info['epoch']} | Loss: {checkpoint_info['loss']:.6f}"
        plt.suptitle(title, fontsize=10, fontweight='bold')
        plt.tight_layout()

        # Save figure
        output_filename = f"generated_samples_epoch_{epoch}.png"
        save_path = output_path / output_filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")

        plt.close()

    print(f"\n{'=' * 60}")
    print(f"✓ All visualizations completed!")
    print(f"{'=' * 60}")

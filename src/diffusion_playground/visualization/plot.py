import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(trajectory: np.ndarray, step: int = 5, title: str = "Forward Diffusion") -> None:
    n_plots = len(trajectory[::step])
    cols = 5
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    counter = 0
    for i, x_t in enumerate(trajectory[::step]):
        axes[i].scatter(x_t[:, 0], x_t[:, 1], s=5)
        axes[i].set_title(f"Step {i * step}")
        axes[i].axis("equal")
        axes[i].axis("off")
        counter += 1

    for i in range(counter, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()


def show_denoising_steps_2d(
        x0: torch.Tensor, xt: torch.Tensor, predicted_noise: torch.Tensor, title: str
) -> None:
    # Calculate the scale of the plot to make all subplots the same size
    x_min, x_max = x0[:, 0].min().item(), x0[:, 0].max().item()
    y_min, y_max = x0[:, 1].min().item(), x0[:, 1].max().item()

    # Add padding
    x_min = min(x_min, x_min * 1.2)
    y_min = min(y_min, y_min * 1.2)
    x_max = max(x_max, x_max * 1.2)
    y_max = max(y_max, y_max * 1.2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x0[:, 0].cpu(), x0[:, 1].cpu(), color='green')
    plt.title("Original")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.subplot(1, 3, 2)
    plt.scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), color='red')
    plt.title("De-Noised (t - 1)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.subplot(1, 3, 3)
    plt.scatter(predicted_noise[:, 0].cpu(), predicted_noise[:, 1].cpu(), color='blue')
    plt.title("Predicted Noise (t)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.suptitle(title, y=0.98, fontsize=12)

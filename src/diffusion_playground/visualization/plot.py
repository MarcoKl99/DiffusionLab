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

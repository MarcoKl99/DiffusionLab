import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.figure

from ..diffusion.noise_schedule import LinearNoiseSchedule
from ..diffusion.backward import generate_samples_conditioned


def evaluate_cifar_10(
        model: nn.Module,
        noise_schedule: LinearNoiseSchedule,
        classes: list[int],
        title: str,
        device: torch.device | str = "cpu",
        num_samples: int = 3,
) -> list[matplotlib.figure.Figure]:
    """
    Generate sample images for each given CIFAR-10 class and return one figure per class.

    :param model: Trained class-conditioned denoiser model
    :param noise_schedule: Noise schedule used during training
    :param classes: List of class indices to evaluate (e.g. [0, 1, 3])
    :param title: Base title shown on every figure (e.g. model name or experiment tag)
    :param device: Device to run generation on
    :param num_samples: Number of images to generate per class (shown as a row)
    :return: One figure per class, in the same order as `classes`
    """

    # Set the class-idx to class-name mapping
    class_idx_to_class_name = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    figures = []

    for class_idx in classes:
        class_label = class_idx_to_class_name[class_idx]
        print(f"Generating {num_samples} sample(s) for class {class_idx} ({class_label})...")

        # Build a label tensor with the same class repeated for every sample
        class_labels = torch.full((num_samples,), class_idx, dtype=torch.long)

        images = generate_samples_conditioned(
            model=model,
            noise_schedule=noise_schedule,
            image_shape=(3, 32, 32),
            class_labels=class_labels,
            device=device,
        )

        # One row of num_samples images
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3.5))
        if num_samples == 1:
            axes = [axes]

        for ax, img in zip(axes, images):
            ax.imshow(img.cpu().numpy())
            ax.axis("off")

        fig.suptitle(f"{title} — {class_label}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        figures.append(fig)

    return figures

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from torchmetrics.image.fid import FrechetInceptionDistance

from ..diffusion.noise_schedule import LinearNoiseSchedule
from ..diffusion.backward import generate_samples_conditioned


class MetricsTracker:
    """
    Tracks training metrics (loss, FID) and generated image samples during diffusion model training.

    Saves metrics to a JSON file and sample grids to PNG files within a `model-metrics`
    subdirectory of the checkpoint directory. Supports visualization of all recorded metrics
    via the `visualize()` method.

    Intended to be called after each checkpoint interval during training, and can also be
    used standalone in a notebook to inspect results after training.
    """

    def __init__(
            self,
            checkpoint_dir: str,
            noise_schedule: LinearNoiseSchedule,
            image_shape: tuple[int, int, int],
            eval_classes: list[int],
            real_images: torch.Tensor,
            num_samples_per_class: int = 4,
            num_fid_samples: int = 1024,
            device: str | torch.device = "cpu",
            class_names: dict[int, str] | None = None,
    ):
        """
        :param checkpoint_dir: Directory where model checkpoints are saved. Metrics are stored
                               in a `model-metrics` subdirectory of this path.
        :param noise_schedule: Noise schedule used during training (reused for sample generation).
        :param image_shape: Shape of generated images as (channels, height, width).
        :param eval_classes: Class indices to generate sample grids for at each evaluation step.
        :param real_images: Full training set tensor in [-1, 1] range, used as the real
                            image reference distribution for FID computation.
        :param num_samples_per_class: Number of images generated per class in the sample grid.
        :param num_fid_samples: Total number of generated images used to compute FID.
                                Distributed evenly across eval_classes.
        :param device: Device to run generation and FID computation on.
        :param class_names: Optional mapping from class index to human-readable name, used
                            for sample grid row labels.
        """

        self.noise_schedule = noise_schedule
        self.image_shape = image_shape
        self.eval_classes = eval_classes
        self.real_images = real_images
        self.num_samples_per_class = num_samples_per_class
        self.num_fid_samples = num_fid_samples
        self.device = torch.device(device) if isinstance(device, str) else device
        self.class_names = class_names or {i: str(i) for i in eval_classes}

        # Directory structure
        self.metrics_dir = Path(checkpoint_dir) / "model-metrics"
        self.samples_dir = self.metrics_dir / "samples"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.metrics_dir / "metrics.json"
        self._metrics = self._load_or_init_metrics()

    def evaluate(self, model: nn.Module, epoch: int, train_loss: float) -> dict:
        """
        Run a full evaluation step: generate a sample grid, compute FID, and persist results.

        The model is temporarily set to eval mode and restored to train mode after evaluation.
        Call this after each checkpoint interval during training.

        :param model: The model being trained.
        :param epoch: Current epoch number (1-indexed, matching checkpoint filenames).
        :param train_loss: Average training loss for this epoch.
        :return: The metrics entry that was appended to the JSON file.
        """

        print(f"\n[MetricsTracker] Evaluating epoch {epoch}...")

        model.eval()
        sample_path = self._generate_sample_grid(model, epoch)
        fid_score = self._compute_fid(model)
        model.train()

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "fid": fid_score,
            "sample_path": str(sample_path.relative_to(self.metrics_dir.parent)),
        }
        self._metrics["entries"].append(entry)
        self._save_metrics()

        print(f"[MetricsTracker] loss={train_loss:.6f} | FID={fid_score:.2f} | saved {sample_path.name}")
        return entry

    def visualize(self, save_path: str | None = None) -> None:
        """
        Plot training loss and FID score as two separate figures.

        :param save_path: Directory to save the figures into. When provided, saves
                          `training_loss.png` and `fid.png` to that directory instead
                          of displaying them. The directory is created if it does not exist.
        """

        entries = self._metrics["entries"]
        if not entries:
            print("No evaluation metrics recorded yet.")
            return

        out_dir = Path(save_path) if save_path else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        epochs = [e["epoch"] for e in entries]
        losses = [e["train_loss"] for e in entries]
        fid_entries = [(e["epoch"], e["fid"]) for e in entries if e.get("fid") is not None]

        # Training-Loss
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, losses, color="steelblue", linewidth=1.5)
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_dir is not None:
            plt.savefig(out_dir / "training_loss.png", dpi=120, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

        # FID
        if not fid_entries:
            return

        fid_epochs, fid_scores = zip(*fid_entries)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fid_epochs, fid_scores, color="tomato", linewidth=1.5, marker="o", markersize=4)
        ax.set_title("FID-Score")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_dir is not None:
            plt.savefig(out_dir / "fid.png", dpi=120, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def _generate_sample_grid(self, model: nn.Module, epoch: int) -> Path:
        """
        Generate `num_samples_per_class` images for each class in eval_classes and save
        as a PNG grid (rows = classes, columns = samples).
        """

        all_labels = [cls for cls in self.eval_classes for _ in range(self.num_samples_per_class)]
        images = generate_samples_conditioned(
            model=model,
            noise_schedule=self.noise_schedule,
            image_shape=self.image_shape,
            class_labels=torch.tensor(all_labels),
            device=self.device,
        )

        n_rows = len(self.eval_classes)
        n_cols = self.num_samples_per_class

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)

        for row, cls in enumerate(self.eval_classes):
            for col in range(n_cols):
                ax = axes[row, col]
                ax.imshow(images[row * n_cols + col].cpu().numpy())
                ax.axis("off")
                if col == 0:
                    ax.text(
                        -0.08, 0.5,
                        self.class_names.get(cls, str(cls)),
                        transform=ax.transAxes,
                        fontsize=9, va="center", ha="right",
                    )

        plt.suptitle(f"Generated Samples — Epoch {epoch}", fontsize=11, fontweight="bold")
        plt.tight_layout()

        path = self.samples_dir / f"epoch_{epoch:06d}.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

        return path

    def _compute_fid(self, model: nn.Module) -> float:
        """
        Compute FID between a random subset of real training images and generated images.

        Real images are converted from [-1, 1] to [0, 1] before InceptionV3 feature extraction.
        Generated images from `generate_samples_conditioned` are already in [0, 1].
        """

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # Real images: random subset, convert [-1, 1] → [0, 1]
        indices = torch.randperm(len(self.real_images))[:self.num_fid_samples]
        real_batch = self.real_images[indices].to(self.device)
        real_batch = ((real_batch + 1) / 2).clamp(0, 1)
        fid.update(real_batch, real=True)

        # Generated images: evenly distributed across eval_classes
        samples_per_class = max(1, self.num_fid_samples // len(self.eval_classes))
        all_labels = [cls for cls in self.eval_classes for _ in range(samples_per_class)]
        images = generate_samples_conditioned(
            model=model,
            noise_schedule=self.noise_schedule,
            image_shape=self.image_shape,
            class_labels=torch.tensor(all_labels),
            device=self.device,
        )

        # Stack list of (H, W, C) → (N, C, H, W)
        fake_batch = torch.stack([img.permute(2, 0, 1) for img in images]).to(self.device)
        fid.update(fake_batch, real=False)

        return fid.compute().item()

    def _load_or_init_metrics(self) -> dict:
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                print(f"Loaded metrics from {self.metrics_path}")
                return json.load(f)
        return {"entries": []}

    def _save_metrics(self) -> None:
        with open(self.metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2)

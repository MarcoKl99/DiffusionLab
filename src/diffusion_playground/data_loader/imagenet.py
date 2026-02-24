from pathlib import Path

import torch
from torchvision import transforms, datasets
from datasets import load_dataset as hf_load_dataset


def load_imagenet(split: str = "train", path_data: str = "data/tiny-imagenet", limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Load Tiny ImageNet dataset. Loads from a local ImageNet-style directory if available,
    otherwise downloads from HuggingFace, saves locally, and then loads.

    Local directory structure:
        <path_data>/
        ├── train/
        │   ├── n01443537/
        │   │   ├── 000000.JPEG
        │   │   └── ...
        │   └── ...
        └── valid/
            ├── n01443537/
            └── ...

    This structure is compatible with torchvision.datasets.ImageFolder and
    can be swapped for full ImageNet without any code changes.

    Tiny ImageNet: 200 classes, 64x64 RGB images.
    100,000 training samples, 10,000 validation samples.

    :param split: Dataset split to load ("train" or "valid").
    :param path_data: Root directory of the local ImageNet-style cache.
    :param limit: Optional max number of samples to load. Useful for fast local runs.
    :return: Tuple of (images [N, 3, 64, 64], labels [N], class_idx_to_name dict).
    """
    split_dir = Path(path_data) / split

    if not split_dir.exists() or not any(split_dir.iterdir()):
        print(f"No local cache found. Downloading Tiny ImageNet ({split}) from HuggingFace...")
        save_imagenet(split=split, path_data=path_data)

    return _load_from_image_folder(split_dir, limit=limit)


def save_imagenet(split: str = "train", path_data: str = "data/tiny-imagenet") -> None:
    """
    Download Tiny ImageNet from HuggingFace and save it locally in ImageNet directory structure.

    Each image is saved as a JPEG under <path_data>/<split>/<class_name>/<idx>.JPEG,
    which is the standard ImageNet layout understood by torchvision.datasets.ImageFolder.

    :param split: Dataset split to download and save ("train" or "valid").
    :param path_data: Root directory to save the dataset into.
    """
    cache_dir = Path(path_data)
    print(f"Downloading Tiny ImageNet ({split}) from HuggingFace...")
    dataset = hf_load_dataset("zh-plus/tiny-imagenet", split=split)
    class_names = dataset.features["label"].names

    for idx, example in enumerate(dataset):
        class_name = class_names[example["label"]]
        class_dir = cache_dir / split / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        example["image"].convert("RGB").save(class_dir / f"{idx:06d}.JPEG")

    print(f"Tiny ImageNet ({split}) saved to {cache_dir / split}")


def _load_from_image_folder(split_dir: Path, limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load an ImageNet-style directory into tensors using ImageFolder."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB channels to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    class_idx_to_name = {v: k for k, v in dataset.class_to_idx.items()}

    n = min(limit, len(dataset)) if limit is not None else len(dataset)
    print(f"Loading Tiny ImageNet from local cache: {split_dir} ({n} samples)")
    images = torch.stack([dataset[i][0] for i in range(n)])
    labels = torch.tensor([dataset[i][1] for i in range(n)])

    return images, labels, class_idx_to_name

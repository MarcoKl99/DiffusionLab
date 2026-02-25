from pathlib import Path

import torch
from torchvision import transforms, datasets
from datasets import load_dataset as hf_load_dataset


def load_imagenet(
        split: str = "train",
        path_data: str = "data/tiny-imagenet",
        limit: int | None = None,
        class_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Load Tiny ImageNet dataset. Loads from a local ImageNet-style directory if available,
    otherwise downloads from HuggingFace, saves locally, and then loads.

    :param split: Dataset split to load ("train" or "valid").
    :param path_data: Root directory of the local ImageNet-style cache.
    :param limit: Optional max number of samples to load. Useful for fast local runs.
    :param class_indices: Optional list of class indices to load (e.g. list(range(10)) for
                          the first 10 classes). Class indices follow ImageFolder's alphabetical
                          ordering of the synset directories. If None, all classes are loaded.
    :return: Tuple of (images [N, 3, 64, 64], labels [N], class_idx_to_name dict).
    """
    split_dir = Path(path_data) / split

    if not split_dir.exists() or not any(split_dir.iterdir()):
        print(f"No local cache found. Downloading Tiny ImageNet ({split}) from HuggingFace...")
        save_imagenet(split=split, path_data=path_data)

    return _load_from_image_folder(split_dir, limit=limit, class_indices=class_indices)


def save_imagenet(split: str = "train", path_data: str = "data/tiny-imagenet") -> None:
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


def _load_from_image_folder(
        split_dir: Path,
        limit: int | None = None,
        class_indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load an ImageNet-style directory into tensors using ImageFolder."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB channels to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root=str(split_dir), transform=transform)
    class_idx_to_name = {v: k for k, v in dataset.class_to_idx.items()}

    # Filter to the requested classes before touching any image files.
    # dataset.samples is a list of (path, class_idx) built from the directory
    # scan — no pixel data has been read at this point.
    samples = dataset.samples
    if class_indices is not None:
        class_set = set(class_indices)
        samples = [(path, idx) for path, idx in samples if idx in class_set]
        class_idx_to_name = {i: class_idx_to_name[i] for i in class_indices if i in class_idx_to_name}

    n = min(limit, len(samples)) if limit is not None else len(samples)
    print(f"Loading Tiny ImageNet from local cache: {split_dir} ({n} samples, {len(class_idx_to_name)} classes)")

    images, labels = [], []
    for path, class_idx in samples[:n]:
        img = dataset.transform(dataset.loader(path))
        images.append(img)
        labels.append(class_idx)

    return torch.stack(images), torch.tensor(labels), class_idx_to_name

import torch
from torchvision import datasets, transforms


def load_cifar_10(path_data: str = "data", download: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper to load transformed CIFAR-10 dataset for training.

    :param path_data: Path to which the dataset should be saved, if not on disk already.
    :param download: Boolean, if the dataset should be downloaded if not present yet.
    :return: Tuple containing the data (images) and labels (classes).
    """

    # Load CIFAR-10 dataset directly as a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to [-1, 1]
    ])

    cifar_dataset = datasets.CIFAR10(root=path_data, train=True, transform=transform, download=download)

    # Extract all images into a single tensor
    cifar_data = torch.stack([cifar_dataset[i][0] for i in range(len(cifar_dataset))])
    cifar_labels = torch.tensor([cifar_dataset[i][1] for i in range(len(cifar_dataset))])

    return cifar_data, cifar_labels

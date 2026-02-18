import torch
from torchvision import datasets, transforms


def load_cifar_10() -> tuple[torch.Tensor, torch.Tensor]:
    # Load CIFAR-10 dataset directly as a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels to [-1, 1]
    ])

    cifar_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=False)

    # Extract all images into a single tensor
    cifar_data = torch.stack([cifar_dataset[i][0] for i in range(len(cifar_dataset))])
    cifar_labels = torch.tensor([cifar_dataset[i][1] for i in range(len(cifar_dataset))])

    return cifar_data, cifar_labels

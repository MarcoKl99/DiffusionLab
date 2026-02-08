from sklearn import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int = 128, download: bool = True) -> tuple[DataLoader, tuple[int, int, int]]:
    # Create a transformation pipeline (tensor + normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the mnist dataset
    mnist_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=download)
    data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return data_loader, (1, 28, 28)

from sklearn import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int = 128) -> tuple[DataLoader, tuple[int, int, int]]:
    """
    Load the MNIST data and create and return a data loader instance based on the given batch size.

    :param batch_size: Batch size to use for the created data loader instance
    :return: Tuple containing
                - Data loader for the MNIST dataset, using the given batch size
                - Shape of each element in the provided MNIST data batches
    """

    # Create a transformation pipeline (tensor + normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the mnist dataset
    mnist_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    data_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return data_loader, (1, 28, 28)

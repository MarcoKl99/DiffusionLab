import numpy as np
from sklearn import datasets


def make_moons_dataset(n_samples: int = 500, noise: float = 0.05) -> np.ndarray:
    data, _ = datasets.make_moons(n_samples=n_samples, noise=noise)
    return data.astype(np.float32)


def make_swiss_roll_dataset(n_samples: int = 500, noise: float = 0.5) -> np.ndarray:
    data, _ = datasets.make_swiss_roll(n_samples=n_samples, noise=noise)
    data = data[:, [0, 2]]
    return data.astype(np.float32)


def make_gaussian_mixture(n_samples: int = 500, centers: int = 4, std: float = 0.1) -> np.ndarray:
    data, _ = datasets.make_blobs(n_samples=n_samples, centers=centers, cluster_std=std)
    return data.astype(np.float32)


def load_toy_dataset(name: str, **kwargs) -> np.ndarray:
    if name == "moons":
        return make_moons_dataset(**kwargs)
    elif name == "swiss_roll":
        return make_swiss_roll_dataset(**kwargs)
    elif name == "gaussian":
        return make_gaussian_mixture(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

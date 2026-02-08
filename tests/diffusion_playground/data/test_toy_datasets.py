import pytest
import numpy as np

from src.diffusion_playground.data_loader.toy_datasets import make_moons_dataset
from src.diffusion_playground.data_loader.toy_datasets import make_swiss_roll_dataset
from src.diffusion_playground.data_loader.toy_datasets import make_gaussian_mixture
from src.diffusion_playground.data_loader.toy_datasets import load_toy_dataset


def test_make_moons_dataset():
    dataset = make_moons_dataset()
    assert isinstance(dataset, np.ndarray)


def test_make_swiss_roll_dataset():
    dataset = make_swiss_roll_dataset()
    assert isinstance(dataset, np.ndarray)


def test_make_gaussian_mixture_dataset():
    dataset = make_gaussian_mixture()
    assert isinstance(dataset, np.ndarray)


def test_load_toy_dataset():
    names = ["moons", "swiss_roll", "gaussian"]
    for name in names:
        dataset = load_toy_dataset(name)
        assert isinstance(dataset, np.ndarray)

    with pytest.raises(ValueError):
        _ = load_toy_dataset("non-existing-dataset-name")

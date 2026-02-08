# Data 📈 Package – Toy Datasets for Diffusion Playground

This package provides simple, generic **2D toy datasets** that we use for developing and testing diffusion models.  
The idea is to first use small, easy-to-understand datasets before scaling up to real image data like MNIST or CIFAR.

---

## Motivation 🏆

- **Quick experimentation:** Small datasets train immediately and are easy to visualize.
- **Easy debugging:** Errors in the diffusion process become immediately visible.
- **Modular extendability:** New datasets can be added easily.
- **Educational goal:** Intuitive understanding of the forward and reverse diffusion process.

---

## Included Datasets ❓

1. **Moons**

```python
make_moons_dataset(n_samples=500, noise=0.05)
```

- Two interleaving half moons
- `noise` controls the spread of the moons
- Classic 2D dataset used in statistics and ML literature

2. **Swiss Roll**

```python
make_swiss_roll_dataset(n_samples=500, noise=0.05)
```

- Spiral like 3D structure, projected to 2D
- Ideal for visualizing complex data structures
- `noise` adds random pertubation

3. **Gaussian Mixture**

```python
make_gaussian_mixture(n_samples=500, centers=4, std=0.1)
```

- Multiple Gaussian clusters in 2D
- `centers` = number of clusters
- `std` = standard deviation of points within a cluster

## Dataset Factory 🏭

For convenient usage, there is the function:

```python
load_toy_dataset(name: str, ** kwargs) -> np.ndarray
```

Examples:

```python
from src.diffusion_playground.data_loader.toy_datasets import load_toy_dataset

X = load_toy_dataset("moons", n_samples=1000, noise=0.1)
X = load_toy_dataset("swiss_roll", n_samples=800)
X = load_toy_dataset("gaussian", n_samples=500, centers=3)
```

Note: `name`must be one of `moons`, `swiss_roll`, or `gaussian`, additional parameters are
passed directly to the respective dataset function.

## Visualization 🔎

To visualize the data, follow the example notebook under `notebooks/toy_dataset_exploration.ipynb`.

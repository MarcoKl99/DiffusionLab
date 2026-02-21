# DiffusionLab 🧠🧪

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An implementation of a generic model structure for **Denoising Diffusion Probabilistic Models (DDPM)** under
variable conditioning.

This project provides the basic structure of a diffusion process for both training and inference, using a generic
diffusion-wrapper. We can inject a U-Net backbone into the wrapper to power the process and provide variable
conditioning (e.g. time + class label) to steer the generation process. The implementation also shows a few examples
on how the structure can be utilized, explaining experiments on various datasets with different models
provided in the package.

All experiments can be replicated using the given notebooks under `notebooks/experiments`.

## 🎯 Project Highlights

- **Generic Training Pipeline**: One unified training function works across toy datasets (2D Moons), grayscale images (
  MNIST), and color images (CIFAR-10) with various backbone models, as long as the generic interface
  `src/diffusion_playground/models/backbones/base_backbone.py` is implemented
- **Multiple Architectures**: From simple MLPs to U-Net-style CNNs with time- and label-embeddings

## 🔬 What's inside

The `diffusion_playground` package provides complete functionality for:

- **Model Architecture**: Generic interface together with sample implementations (based on the U-Net architecture)
- **Reverse Diffusion**: Train models to denoise and generate new samples
- **Training Infrastructure**: Checkpointing, resumption, and progress tracking

Together with the implementations, the `README` files show some basic mathematical ideas
of the diffusion process.

## Contents 📖

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Installation 🚀

To install the project, follow the steps outlines below. Note that this is a basic Python
setup and can therefore easily be extended to further package managers etc.

1. Clone the repository
2. Create a virtual environment for Python

```bash
python3 -m venv .venv
```

3. Install the required packages

```bash
pip install -r requirements.txt
```

With these steps, the setup is completed and the given scripts as well as notebooks
can be executed.

## Usage ⚙️

### Package Structure

The `diffusion_playground` package is organized into several key modules, the most
important parts of which can be seen below.

```
diffusion_playground/
├── diffusion/                                # Core diffusion algorithms
│   ├── noise_schedule.py                     # Linear noise scheduling (β, α, ᾱ)
│   ├── training_utils.py                     # Forward diffusion (add noise)
│   └── backward.py                           # Reverse diffusion (generate samples)
├── models/                                   # Neural network architectures
│   ├── backbones                             # Base-models that power the diffusion process
│   ├── cond_embedding.py                     # Different ways for conditional embedding classes
│   ├── time_and_class_conditioned_model.py   # Wrapper for models steerable by the class label
│   └── time_conditioned_model.py             # Wrapper for models without class label input during generation
└── training/                                 # Training infrastructure
    └── denoiser_trainer.py                   # Generic training loop with checkpointing
```

### Quick Start

To start an experiment, simply follow the implementations under `notebooks/experiments`
and e.g. train a diffusion model on the CIFAR-10 dataset.

To not constantly change this README, no code-snippets are provided here... these
change about as quickly as the most popular disruptive agentic AI tools for
increasing shareholder value 😉.

### Key Features

- **Automatic Checkpointing**: Models are saved at regular intervals with epoch and loss tracking
- **Seamless Resume**: Training automatically resumes from the latest checkpoint
- **Generic Interface**: Same training function works for any dataset/architecture combination
- **Flexible Sampling**: Easy sampling of generated data using the provided functions

## Examples 📺

This project includes three progressive experiments that demonstrate the versatility and power of our diffusion model
implementation:

### CIFAR-10 Object Generation 🎨

**Goal**: Scale to complex color images with multiple object categories, while conditioning the generation process on
the class labels. This enables us to specifically generate e.g. a horse, a car, etc. by providing the corresponding
label as an input.

- **Architecture**: U-Net CNN versions (3M and 53M parameters), Sinusoidal time embedding + class label embedding
- **Dataset**: 50,000 color images (32×32 RGB) across 10 object classes
- **Training**: 100,000 epochs
- **Results**: Learns object shapes and structures, starts to be recognizable after 100,000 epochs

**Examples:**

...

**Key Observations**:

- ✅ Shape learning is evident
- ✅ Color distributions match object categories (blue for water/ships, browns for animals)
- ✅ Spatial coherence (objects have proper structure and backgrounds)
- ⚠️ Textures are abstract - not realistic yet (requires larger models/more training)

**Documentation**: [CIFAR-10 Experiment README](notebooks/experiments/cifar-10/README.md)

### Running the Experiments

Each experiment is contained in a Jupyter notebook:

```bash
# Activate your environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook

# Navigate to:
# - notebooks/experiments/moons/moons_diffusion.ipynb
# - notebooks/experiments/mnist/mnist_diffusion.ipynb
# - notebooks/experiments/cifar-10/cifar10_diffusion.ipynb
```

Each notebook includes:

- Dataset loading (and exploration in relevant cases)
- Model architecture setup
- Training with automatic checkpointing
- Visualization of results
- Sample generation from checkpoints

## License 🛡️

This project is under the MIT-license. For more information, see [LICENSE](LICENSE).

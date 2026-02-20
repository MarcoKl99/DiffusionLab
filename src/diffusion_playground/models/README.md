# Models - Where intelligence is created! 🧠

This package holds the implementation of the models used to train to generate new data.

## MLP Denoiser 💪

This simple model serves as a PoC (Proof of Concept) for the entire process, as it is a small
implementation of a simple feed forward network, fast to train and evaluate. Using this
we can answer the question:

*Does this even work?* 🤔

The architecture of the implemented model looks as follows.

```text
Linear (input dimension + 1 for the time step)
ReLU
Linear (hidden dimension)
ReLU
Linear (input dimension)  # == output dimension to create the same shape again
```

The model derives from the `nn.Module` and implements a simple `forward` method,
concatenating the input data with the given time step and returning the forward-pass
of the model.

## CNNDenoiser 🤖

This is the first (and smallest) CNN-based model, following the idea of the U-Net
architecture to take noise as an input and generate an image as an output.

The architecture roughly looks as follows:

<img src="../../../docs/models/cnn-denoiser/schema.svg">

(full model graph [here](../../../docs/models/cnn-denoiser/CNNDenoiserGraph.png))

### Properties ❓

- 3,321,475 parameters
- 64 base channels (after init convolution)
- 2 down-/up-sampling blocks each
- Sinusoidal time embeddings

### Sinusoidal Time Embeddings - What is this even? 💭

Great question! 😉 Generally we want to embed the time dimension to be able to work with a more continuous
representation than plain numbers like 1, 2, 3, ...

So we choose to embed the time step into a higher dimensional space for the model to
understand, that similar time steps in that higher dimensional space are closer to each
other. In our case, this embedding dimensionality is

$$
t_{embed} = 128
$$

**Sinus... I know this from triangles! 📐**

The idea is to map a number $ t $ to a vector

$$
PE(t) = [sin(\omega_1 t), sin(\omega_2 t), ..., cos(\omega_1 t), cos(\omega_2 t), ...]
$$

with frequencies

$$
\omega_k = 10,000^{-\frac{k}{d/2 - 1}}
$$

and each sin-/cos-block going up to $ d / 2 $. Note, that these frequencies are
exponentially decreasing and the resulting embedding has a dimension of $ d $.

Because of the fixed nature of this embedding, in many cases the sinusoidal
representation is extended by a small MLP, which is realized with a simple
linear layer in this example.

```python
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        ...

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
```

## CNNDenoiserLarge 🤖🤖

Even though the cryptic name, this is really just an implementation of the basic
CNNDenoiser as described above, just with a slightly larger architecture

### Adaptions compared to the CNNDenoiser 🔧

- Base Channels: 128 (instead of 64)
- Number of down-/up-sampling blocks each 3 (instead of 2)
- Number of parameters 53,441,795 (instead of 3,321,475)

<img src="../../../docs/models/cnn-denoiser-large/schema.svg">

(detailed model graph [here](../../../docs/models/cnn-denoiser-large/CNNDenoiserLargeGraph.png))

The rest of the model remained unchanged, simply exploring the effects of a larger
model using the same architecture and training for the same time on the same dataset.

# Models - Where intelligence is created! 🧠

This package holds the implementation of the models used to train to generate new data.

## MLP Denoiser 💪

This simple model serves as a PoC (Proof of Concept) for the entire process, as it is a small
implementation of a simple feed forward network, fast to train and evaluate. Using this
we can answer the question:

*Does this even work?* 🤔

The architecture of the implemented `Sequential` model looks as follows.

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

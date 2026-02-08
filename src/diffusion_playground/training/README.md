# Training

This package implements the utilities for executing the training loop for a given model,
data, and noise schedule.

## denoiser_trainer.py - or as I call him, "Coach" 💪

The function `train_denoiser` defined in this file performs all prerequisite steps as well
as iterating through the training loop of a model.

Steps 🚶‍♂️‍➡️:

1. Define the Adam-optimizer
2. Send the model to the correct device (GPU if available, else CPU)
3. Iterate
   1. Sample a noised datapoint at randomly chosen time step $ t $
   2. Predict the noise that was added to this time step compared to the previous one
   3. Calculate the loss against the truly added noise
   4. Calculate backward gradients and perform optimizer step

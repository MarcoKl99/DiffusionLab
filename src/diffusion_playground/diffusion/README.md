# Diffusion - Uuhhh sounds like AI hmm? 😏

This module provides functionality around the noising process of data, which enables
us to explore the idea as well as to sample datapoints from a noising-schedule later
in the training.

## Forward Process ➡️

We start with real data, e.g. an image without noise.

$$
x_0 \sim p_{data}
$$

The strategy is to iteratively add Gaussian noise by applying

$$
x_t = \sqrt{1 - \beta} x_{t - 1} + \sqrt{\beta}\epsilon
$$

with $ \epsilon \sim N(0, I) $ for a number of time steps $ t $ so that we and up with

$$
\{x_o, x_1, ..., x_T\}
$$

and

$$
x_T \approx N(0, I)
$$

converging towards pure Gaussian noise. Later we will attempt to learn to reverse that
process with the Diffusion Model.

## Noise Schedule ⏰

This clas serves as a provider of required parameters for the added noise at each
step $ t $ throughout the forward process. The individual parts will be explained briefly.

### $ \beta_t $ - Variance of the added noise

The class defines

```python
self.betas = torch.linspace(beta_start, beta_end, time_steps)
```

as a linear sequence of values with a given start- and endpoint as well as a number
of steps.

When looking at the above described noising process we see, that by applying

$$
x_t = \sqrt{1 - \beta} x_{t - 1} + \sqrt{\beta}\epsilon
$$

the value $ \beta $ defines the strength of the noise added to the original data.
Therefore, a large value for $ \beta $ leads to a significant distortion while
smaller values barely add some noise.

### $ \alpha_t $ - Signal Retention Factor

The values for $ \alpha_t $ at each time step defines the portion of the signal
of the original data that should be kept and is therefore simply calculated by

$$
\alpha_t = 1 - \beta_t
$$

### $ \bar{\alpha}_t $ - Cumulative Signal Retention Factor

Mathematically it can be shown, that the entire forward process has a closed form of

$$
q(x_t|x_0) = N(\sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)
$$

with

$$
\bar{\alpha}_t = \prod_{s=1}^{t}{\alpha_t}
$$

which allows us to implement the forward sampling process without iterating through
all steps of $ t $.

## Training Utils 💪

This file provides a utility function `sample_xt` that allows us to conveniently sample
noised data at a given time step $ t $ from a linear noising schedule and an input $ x_0 $.

The function performs the following steps.

1. Get a random time step based on the possible steps of the noise schedule
2. Get the corresponding $ \bar{\alpha}_t $ from the noise schedule
3. Create Gaussian noise
4. Add the noise based on the given $ \bar{\alpha}_t $ to the data
5. Return the noised data, the added noise, and the corresponding time step $ t $

# CHANGELOG

<!-- version list -->

## v1.1.0 (2026-04-15)

### Bug Fixes

- Fix typo in main notebook of the cifar experiments
  ([`6c7a8bc`](https://github.com/MarcoKl99/DiffusionLab/commit/6c7a8bc9b545001f147220a49e18f9148fb04be1))

- Implement smaller learning rate for larger models, AdamW with Weight Decay, gradient clipping,
  infinite gradient skipping
  ([`8155ed8`](https://github.com/MarcoKl99/DiffusionLab/commit/8155ed896fa3b9018c1406bd04e0c432d4976bad))

### Documentation

- Adapt project README
  ([`8feb066`](https://github.com/MarcoKl99/DiffusionLab/commit/8feb066c46de267ac556736fa455b2e246aadf39))

- Add CNNDenoiser15 Results to README
  ([`8b4a0b3`](https://github.com/MarcoKl99/DiffusionLab/commit/8b4a0b31d34e39fcb9b44da75ac5a4b3be52170d))

- Add results for the CNNDenoiser21 EMA Cosine Schedule run
  ([`14be61e`](https://github.com/MarcoKl99/DiffusionLab/commit/14be61eff256939c15e52a726e7e79b094c01551))

- Added evaluation results
  ([`db82d36`](https://github.com/MarcoKl99/DiffusionLab/commit/db82d36e115776f5fbc7ea1a7e3bc5d40fabf7a6))

- Added results for the CNNDenoiser15 experiment with the cosine noise schedule
  ([`3baac89`](https://github.com/MarcoKl99/DiffusionLab/commit/3baac89a9f5a1f8837eeaa4fe14a59e292477224))

- Correct number of parameters for 21 model, add adapted batch size for training config for 21 model
  onwards
  ([`3ff3fcc`](https://github.com/MarcoKl99/DiffusionLab/commit/3ff3fccb18c52dd475d1d7935ee7d7a4a4ba1f76))

- Correct time duration in docs images
  ([`c72d403`](https://github.com/MarcoKl99/DiffusionLab/commit/c72d4031984ce3c88150c2d2b2fdc962e77d2d62))

- Fix typo in 58M + Attention image
  ([`7a88ded`](https://github.com/MarcoKl99/DiffusionLab/commit/7a88dedbef3d3898a1ecdb85697c83e98a8b73f9))

- Remove removed table of contents heading from project readme
  ([`2f42603`](https://github.com/MarcoKl99/DiffusionLab/commit/2f42603335a225cbd6b9948801aa560c069a966e))

- Results of CNNDenoiserLargeAttention (58M)
  ([`4ad3a56`](https://github.com/MarcoKl99/DiffusionLab/commit/4ad3a5698bd5500b5982b71d7bb724161b7e253d))

### Features

- Add batch-wise FID score calculation to enable larger sample sizes
  ([`5aa67b9`](https://github.com/MarcoKl99/DiffusionLab/commit/5aa67b988894c4e7d6c8084db729cebbbdeba58d))

- Add cosine noise schedule, add docs for the CNNDenoiser15 + EMA experiment
  ([`9e41a07`](https://github.com/MarcoKl99/DiffusionLab/commit/9e41a078b10dc0f43944378690da68a9dedb00c7))

- Add evaluation function for the models trained on the CIFAR dataset (class conditioned) -
  attention based cnndenoiserlargeattention works great
  ([`0d27999`](https://github.com/MarcoKl99/DiffusionLab/commit/0d279995805ad90464929731acbd86f0816ef488))

- Add metrics tracker to training process - save training loss and calculate FID per
  checkpoint-interval
  ([`38c8558`](https://github.com/MarcoKl99/DiffusionLab/commit/38c85589eebf03b709c669a0d84bafc67f3ead6b))

- Add optional manual evaluation cell in main notebook for the cifar experiment, reduce number of
  epochs until output during inference from 100 to 10, implement distinction between
  checkpoint_every and eval_every during training
  ([`64d065a`](https://github.com/MarcoKl99/DiffusionLab/commit/64d065a74089cf74afc0a5d02f48cf0028cc69b1))

- Add start of implementation for tiny-imagenet diffusion - load dataset
  ([`67e07a3`](https://github.com/MarcoKl99/DiffusionLab/commit/67e07a38bb400a55329b62b1da636a634e26f816))

- Implement attention based model (derived from CNNDenoiserLarge), change batch norm to be group
  norm to not mix noise states during normalization
  ([`5d7fec4`](https://github.com/MarcoKl99/DiffusionLab/commit/5d7fec47e0207cd7dd4c9ce6c7bf61c9def25d3c))

- Implement CNNDenoiser15 as a backbone with 15 intermediary Conv blocks
  ([`3e01337`](https://github.com/MarcoKl99/DiffusionLab/commit/3e01337e8a542c9dd621736ee30bda60087dbb5c))

- Implement CNNDenoiserXLAttention (129M) as the by now largest model - implement experiment project
  for Tiny-ImageNet dataset
  ([`4ca0771`](https://github.com/MarcoKl99/DiffusionLab/commit/4ca0771d0cf1b394ed3e8789fb8c6687db077807))

- Implement deeper 21 and 27 models, implement SiLU and down-/up-block residual connections
  ([`16dfe50`](https://github.com/MarcoKl99/DiffusionLab/commit/16dfe50b73d341c4195a0b84c0577e8ca2756441))

- Implement EMA logic for evaluation
  ([`fcc5ee6`](https://github.com/MarcoKl99/DiffusionLab/commit/fcc5ee6846c1f96137bfe13f40d97066b2ea316b))

- Implement purging process when saving model checkpoints to not run into storage issues (especially
  on Google Drive xD)
  ([`2c35725`](https://github.com/MarcoKl99/DiffusionLab/commit/2c35725b3549ea796da29c33c8acacd14c7de7db))

- Move the tqdm to the inner loop (over the entire dataset in batches) to see a better (more
  dynamic) output during training
  ([`57c5719`](https://github.com/MarcoKl99/DiffusionLab/commit/57c57196002d57e9c6c0568b25bd4461d20fe80c))

### Refactoring

- Change metrics tracker directory structure
  ([`329f44d`](https://github.com/MarcoKl99/DiffusionLab/commit/329f44d69aecf5660e09f04e3d7c06ee5f98d826))


## v1.0.0 (2026-02-22)

- Initial Release

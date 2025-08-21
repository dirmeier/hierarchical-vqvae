# Hierarchical VQ-VAE

[![ci](https://github.com/dirmeier/hierarchical-vqvae/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/hierarchical-vqvae/actions/workflows/ci.yaml)

## About

This repository implements a hierarchical three-level VQ-VAE which has been proposed in [Generating Diverse High-Fidelity Images with VQ-VAE](https://arxiv.org/abs/1906.00446) using JAX and Flax.

> [!WARNING]
> The implementation (or maybe the hierarchical VQ-VAE) seems fairly sensitive to initialization. With a random seed of 1 (i.e., `config.rng_key=1`) the training is stable and converges
> after ten epochs (at least on a Nvidia V100). With some other seeds the loss might diverge towards infinity. This behaviour is the same between a ResNetV1 block and a
> ConvNext block.

## Example usage

The `experiments` folder contains a use case on CIFAR10. To run the experiments, first download the latest release
and install all dependencies via:

```bash
wget -qO- https://github.com/dirmeier/hierarchical-vqvae/archive/refs/tags/<TAG>.tar.gz | tar zxvf -
uv sync --all-groups
```

To train a model, just execute:

```bash
cd experiments/cifar10
python main.py
  --config=config.py
  --workdir=<dir>
  (--usewand)
```

Below are reconstructed images from the VQ-VAE using a ConvNext residual block.

<div align="center">
  <img src="experiments/cifar10/figures/reconstructed_images.png" width="750">
</div>

## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/hierarchical-vqvae@<TAG>
```

## Author

Simon Dirmeier <a href="mailto:simd23@pm.me">simd23 @ pm dot me</a>

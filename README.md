# VQ-VAE

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![arXiv](https://img.shields.io/badge/arXiv-2311.00474-b31b1b.svg)](https://arxiv.org/abs/2311.00474)
[![ci](https://github.com/dirmeier/vqvae/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/vqvae/actions/workflows/ci.yaml)
# Denoising diffusion operators

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/denoising-diffusion-operators/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/denoising-diffusion-operators/actions/workflows/ci.yaml)

> Implementation of 'Score-based Diffusion Models in Function Space'

## About

This repository implements the method, denoising diffusion operator (DDO), proposed in [Score-based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400), i.e.,
a function-space version of diffusion probabilistic models, using JAX and Flax.

> [!IMPORTANT]
> The implementation does not strictly follow the experimental setup in the paper (since the paper itself uses a different one for each experiment).
> Specifically, the U-net neural operator ([U-NO](https://arxiv.org/abs/2204.11127)) as well as the sampling are customized and simplified.
> Our U-NO implementation just uses spectral convolutions for up- and down-sampling of input dimensions.
> We use the VP-parameterization of [DDPM](https://arxiv.org/abs/2006.11239); hence we don't use the score-matching loss in [NCSN](https://arxiv.org/abs/1907.05600) but a conventional SSE loss.
> We consequently also don't use Langevin dynamics for sampling, but the sampling proposed in [DDIM](https://arxiv.org/abs/2010.02502).
>
> If you find bugs, please open an issue and report them.

## Example usage

The `experiments` folder contains a use case on MNIST-SDF. For training on 32x32-dimensional images from the MNIST-SDF dataset, call:

```bash
cd experiments/mnist_sdf
python main.py \
  --config=config.py \
  --mode=train \
  --model=<uno|unet> \
  --dataset=mnist_sdf \
  --workdir=<dir>
```

Then, sample images via:

```bash
cd experiments/mnist_sdf
python main.py \
  --config=config.py \
  --mode=sample \
  --model=<uno|unet> \
  --dataset=mnist_sdf \
  --workdir=<dir>
```

Below are DDIM-sampled images from the DDO when either a UNet or a UNO is used as score model (a DDO with a UNet is just a DDPM). The UNet parameterization yields high-quality results already after
20 epochs or so. The UNO works worse than the UNet when 32x32-dimensional images are sampled and takes significantly longer to train. When sampling 64x64-dimensional images it mainly produces noise.

<div align="center">
  <div>UNet 32x32</div>
  <img src="fig/mnist_sdf-unet-32x32.png" width="750">
</div>

<div align="center">
  <div>UNO 32x32</div>
  <img src="fig/mnist_sdf-uno-32x32.png" width="750">
</div>

<div  align="center">
  <div>UNO 64x64</div>
  <img src="fig/mnist_sdf-uno-64x64.png" width="750">
</div>

## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/ddo@<TAG>
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>

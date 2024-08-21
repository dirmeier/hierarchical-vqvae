from collections.abc import Sequence
from dataclasses import asdict

import jax
import numpy as np
from flax import linen as nn
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class VectorQuantizerConfig:
    n_embedding: int = 64
    embedding_dim: int = 512
    commitment_cost: float = 0.25


@struct.dataclass
class VAEConfig:
    n_channels: int = 128
    n_out_channels: int = 128
    n_layers: int = 2
    model_block_name: str = "convnext"


class _ConvNextBlock(nn.Module):
    """Implements a ConvNext blockm see https://arxiv.org/abs/2201.03545"""

    n_channels: int
    n_layers: int

    @nn.compact
    def __call__(self, inputs, is_training):
        hidden = inputs
        for _ in range(self.n_layers):
            intermediate = nn.Conv(
                self.n_channels,
                kernel_size=(7, 7),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.n_channels,
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )(jax.nn.gelu(hidden))
            intermediate = nn.LayerNorm()(intermediate)
            intermediate = nn.Conv(
                self.n_channels * 4,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )(intermediate)
            intermediate = nn.gelu(intermediate)
            intermediate = nn.Conv(
                self.n_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )(intermediate)
            hidden = hidden + intermediate
        return jax.nn.gelu(hidden)


class _ResNetBlock(nn.Module):
    n_channels: int
    n_layers: int

    def setup(self):
        layers = []
        for i in range(self.n_layers):
            conv3 = nn.Conv(
                self.n_channels * 2,
                kernel_size=(3, 3),
                strides=(1, 1),
                name="res3x3_%d" % i,
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )
            conv1 = nn.Conv(
                self.n_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                name="res1x1_%d" % i,
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )
            layers.append((conv3, conv1))
        self._layers = layers

    def __call__(self, inputs, is_training):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(jax.nn.gelu(h))
            conv1_out = conv1(jax.nn.gelu(conv3_out))
            h += conv1_out
        return jax.nn.gelu(h)


class Encoder(nn.Module):
    n_channels: int
    n_out_channels: int
    n_layers: int
    model_block_name: str

    @nn.compact
    def __call__(self, inputs, is_training):
        block_ctor = (
            _ConvNextBlock
            if self.model_block_name == "convnext"
            else _ResNetBlock
        )
        hidden = inputs
        hidden = nn.Conv(
            self.n_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = nn.Conv(
            self.n_channels,
            kernel_size=(4, 4),
            strides=(1, 1),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = nn.Conv(
            self.n_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)

        hidden = block_ctor(self.n_out_channels, self.n_layers)(
            hidden, is_training
        )
        return hidden


class Decoder(nn.Module):
    n_channels: int
    n_layers: int
    n_out_channels: int
    model_block_name: str

    @nn.compact
    def __call__(self, inputs, is_training):
        block_ctor = (
            _ConvNextBlock
            if self.model_block_name == "convnext"
            else _ResNetBlock
        )
        hidden = inputs
        hidden = nn.Conv(
            self.n_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)
        hidden = block_ctor(self.n_channels, self.n_layers)(hidden, is_training)
        hidden = nn.ConvTranspose(
            self.n_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = nn.Conv(
            self.n_out_channels,
            kernel_size=(4, 4),
            strides=(1, 1),
            kernel_init=nn.initializers.variance_scaling(
                0.1, "fan_in", distribution="uniform"
            ),
            bias_init=nn.initializers.zeros,
        )(hidden)
        return hidden


class _Upsample(nn.Module):
    h_and_w: Sequence[int]
    kernel_size: int = 3

    @nn.compact
    def __call__(self, inputs):
        B, H, W, C = inputs.shape
        n_steps = np.log2(self.h_and_w[0] // H).astype(np.int32)
        hidden = inputs
        for i in range(n_steps):
            hidden = nn.ConvTranspose(
                inputs.shape[-1],
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding="SAME",
                kernel_init=nn.initializers.variance_scaling(
                    0.1, "fan_in", distribution="uniform"
                ),
                bias_init=nn.initializers.zeros,
            )(hidden)
        return hidden


class VectorQuantizer(nn.Module):
    """Adopted from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/nets/vqvae.py#L41"""

    n_embedding: int
    embedding_dim: int
    commitment_cost: float

    def setup(self):
        self.embeddings = self.param(
            "embedding",
            nn.initializers.variance_scaling(
                1.0, "fan_in", distribution="uniform"
            ),
            (self.embedding_dim, self.n_embedding),
        )

    def __call__(self, inputs, **kwargs):
        flat_inputs = jnp.reshape(inputs, [-1, self.embedding_dim])

        distances = (
            jnp.sum(jnp.square(flat_inputs), 1, keepdims=True)
            - 2 * jnp.matmul(flat_inputs, self.embeddings)
            + jnp.sum(jnp.square(self.embeddings), 0, keepdims=True)
        )

        encoding_indices = jnp.argmax(-distances, 1)
        encodings = jax.nn.one_hot(
            encoding_indices, self.n_embedding, dtype=distances.dtype
        )

        encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])
        quantized = self.quantize(encoding_indices)

        e_latent_loss = jnp.mean(
            jnp.square(jax.lax.stop_gradient(quantized) - inputs)
        )
        q_latent_loss = jnp.mean(
            jnp.square(quantized - jax.lax.stop_gradient(inputs))
        )
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
        avg_probs = jnp.mean(encodings, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "loss": loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }

    def quantize(self, encoding_indices):
        w = self.embeddings.swapaxes(1, 0)
        w = jax.device_put(w)
        return w[(encoding_indices,)]


class VQVAE(nn.Module):
    encoder_configs: Sequence[VAEConfig]
    decoder_configs: Sequence[VAEConfig]
    vq_configs: Sequence[VectorQuantizerConfig]

    def setup(self):
        assert len(self.decoder_configs) == len(self.encoder_configs)
        assert len(self.decoder_configs) == len(self.vq_configs)
        self.n_scaling = len(self.decoder_configs)
        self._encoders = [
            Encoder(**asdict(self.encoder_configs[i]))
            for i in range(self.n_scaling)
        ]
        self._decoders = [
            Decoder(**asdict(self.decoder_configs[self.n_scaling - 1 - i]))
            for i in range(self.n_scaling)
        ]
        self._quantizers = [
            VectorQuantizer(**asdict(self.vq_configs[i]))
            for i in range(self.n_scaling)
        ]

    @nn.compact
    def __call__(self, inputs, is_training):
        encs = self._encode(inputs, is_training)
        decs, quants = [], []
        vq_loss = 0
        for idx in list(reversed(range(self.n_scaling))):
            quantizable_input = (
                [encs[idx], decs[-1]] if len(decs) else [encs[idx]]
            )
            pre_quant_conv = self._pre_quantization_conv(
                self.vq_configs[idx].embedding_dim, *quantizable_input
            )
            vq_res = self._quantizers[idx](pre_quant_conv)
            decoded = self._decode(
                idx, vq_res["quantized"], is_training, *quants
            )
            vq_loss += vq_res["loss"]
            quants.append(vq_res["quantized"])
            decs.append(decoded)

        outputs = decs[-1]
        recon_loss = jnp.mean(jnp.square(inputs - outputs))
        loss = recon_loss + vq_loss
        return {"outputs": outputs, "loss": loss}

    def _decode(self, idx, inputs, is_training, *args):
        if len(args) > 1:
            upsampled = self._upsample(args, inputs.shape[1:3])
            inputs = jnp.concatenate([inputs, *upsampled], axis=-1)
        decoded = self._decoders[idx](inputs, is_training)
        return decoded

    def _pre_quantization_conv(self, embedding_dim, *args):
        return nn.Conv(
            embedding_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
        )(jnp.concatenate(args, axis=-1))

    def _encode(self, inputs, is_training):
        encs = [inputs]
        for i in range(self.n_scaling):
            encoded = self._encoders[i](encs[-1], is_training)
            encs.append(encoded)
        encs = encs[1:]
        return encs

    def _upsample(
        self,
        quantizeds,
        shape,
    ):
        upsampleds = []
        for i, quantized in enumerate(quantizeds):
            upsampleds.append(_Upsample(shape[1:3])(quantized))
        return upsampleds

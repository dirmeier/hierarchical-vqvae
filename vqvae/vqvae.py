from collections.abc import Iterable, Sequence

import chex
import jax
from einops import rearrange
from flax import linen as nn
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class VectorQuantizerConfig:
    n_embedding: int = 32
    embedding_dim: int = 32
    commitment_cost: float = 0.25


@struct.dataclass
class VAEConfig:
    n_channels: int = 32
    n_out_channels: int = 32
    channel_multipliers: Sequence[int] = (1, 2)

    n_resnet_blocks: int = 2
    attention_resolutions: Sequence[int] = ()
    n_attention_heads: int = 1

    kernel_size: int = 3
    dropout_rate: float = 0.1

    use_conv_in_resize: bool = True
    n_groups: int = 32


class _ResidualBlock(nn.Module):
    n_out_channels: int
    dropout_rate: float
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, inputs, is_training):
        hidden = inputs
        # convolution with pre-layer norm
        hidden = nn.GroupNorm(num_groups=self.n_groups)(hidden)
        hidden = nn.silu(hidden)
        hidden = nn.Conv(
            self.n_out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding="SAME",
        )(hidden)
        # convolution with pre-layer norm and dropout
        hidden = nn.GroupNorm(num_groups=self.n_groups)(hidden)
        hidden = nn.silu(hidden)
        hidden = nn.Dropout(self.dropout_rate)(hidden, deterministic=not is_training)
        hidden = nn.Conv(
            self.n_out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(1, 1),
            padding="SAME",
        )(hidden)

        if inputs.shape[-1] != self.n_out_channels:
            residual = nn.Conv(
                self.n_out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
            )(inputs)
        else:
            residual = inputs

        return hidden + residual


class _DotProductAttention(nn.Module):
    n_heads: int = 1

    @nn.compact
    def __call__(self, inputs):
        B, H, W, C = inputs.shape
        chex.assert_equal(C % (3 * self.n_heads), 0)
        q, k, v = jnp.split(inputs, 3, axis=3)
        outputs = nn.attention.dot_product_attention(
            rearrange(q, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
            rearrange(k, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
            rearrange(v, "b h w (c heads) -> b (h w) heads c", heads=self.n_heads),
        )
        outputs = rearrange(
            outputs,
            "b (h w) heads c -> b h w (heads c)",
            heads=self.n_heads,
            h=H,
            w=W,
        )
        return outputs


class _AttentionBlock(nn.Module):
    n_heads: int
    n_groups: int

    @nn.compact
    def __call__(self, inputs, is_training):
        hidden = inputs
        hidden = nn.GroupNorm(self.n_groups)(hidden)
        # input projection (replacing the MLP on conventional attention)
        hidden = nn.Conv(
            inputs.shape[-1] * 3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
        )(hidden)
        # attention, we don't push through linear layers since we have the
        # convolution above outputting 3 times the layers which we use as
        # k, q, v
        hidden = _DotProductAttention(self.n_heads)(hidden)
        # output projection (replacing the MLP on conventional attention)
        outputs = nn.Conv(
            inputs.shape[-1],
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(hidden)
        return outputs + inputs


class _Downsample(nn.Module):
    use_conv: bool
    kernel_size: int = 3
    stride: int = 2

    @nn.compact
    def __call__(self, inputs, is_training):
        if self.use_conv:
            outputs = nn.Conv(
                inputs.shape[-1],
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(self.stride, self.stride),
                padding=self.kernel_size // 2,
            )(inputs)
        else:
            outputs = nn.avg_pool(
                inputs,
                window_shape=(self.stride, self.stride),
                strides=(self.stride, self.stride),
            )
        return outputs


class _Upsample(nn.Module):
    h_and_w: Iterable[int]
    use_conv: bool = True
    kernel_size: int = 3
    stride: int = 1

    @nn.compact
    def __call__(self, inputs, is_training):
        B, H, W, C = inputs.shape
        outputs = jax.image.resize(
            inputs,
            (B, *self.h_and_w, C),
            method="nearest",
        )
        if self.use_conv:
            outputs = nn.Conv(
                inputs.shape[-1],
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(self.stride, self.stride),
                padding="SAME",
            )(outputs)
        return outputs


class Encoder(nn.Module):
    config: VAEConfig

    @nn.compact
    def __call__(self, inputs, is_training, **kwargs):
        # the input is assumed to be channel last (as is the convention in Flax)
        # B, H, W, C = inputs.shape
        hidden = inputs
        # lift data
        hidden = nn.Conv(
            self.config.n_channels * self.config.channel_multipliers[0],
            kernel_size=(self.config.kernel_size, self.config.kernel_size),
            strides=(1, 1),
            padding="SAME",
        )(hidden)

        # left block of UNet
        for level, channel_mult in enumerate(self.config.channel_multipliers):
            n_outchannels = channel_mult * self.config.n_channels
            for _ in range(self.config.n_resnet_blocks):
                hidden = _ResidualBlock(
                    n_out_channels=n_outchannels,
                    dropout_rate=self.config.dropout_rate,
                    kernel_size=self.config.kernel_size,
                    n_groups=self.config.n_groups,
                )(hidden, is_training)
                if hidden.shape[-1] in self.config.attention_resolutions:
                    hidden = _AttentionBlock(
                        n_heads=self.config.n_attention_heads,
                        n_groups=self.config.n_groups,
                        use_flash_attention=self.config.use_flash_attention,
                    )(hidden, is_training)
            if level != len(self.config.channel_multipliers) - 1:
                hidden = _Downsample(
                    use_conv=self.config.use_conv_in_resize,
                    kernel_size=self.config.kernel_size,
                )(hidden, is_training)

        outputs = nn.Conv(
            self.config.n_out_channels,
            kernel_size=(self.config.kernel_size, self.config.kernel_size),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(hidden)

        return outputs


class Decoder(nn.Module):
    config: VAEConfig

    @nn.compact
    def __call__(self, inputs, is_training, **kwargs):
        # the input is assumed to be channel last (as is the convention in Flax)
        # B, H, W, C = inputs.shape
        hidden = inputs
        # lift data
        hidden = nn.Conv(
            self.config.n_channels * self.config.channel_multipliers[-1],
            kernel_size=(self.config.kernel_size, self.config.kernel_size),
            strides=(1, 1),
            padding="SAME",
        )(hidden)

        for level, channel_mult in reversed(list(enumerate(self.config.channel_multipliers))):
            n_outchannels = channel_mult * self.config.n_channels
            if level != len(self.config.channel_multipliers) - 1:
                n_outchannels = channel_mult * self.config.n_channels
                hidden = _Upsample(
                    (hidden.shape[1] * 2, hidden.shape[2] * 2),
                    kernel_size=self.config.kernel_size,
                )(hidden, is_training)
            for _ in range(self.config.n_resnet_blocks):
                hidden = _ResidualBlock(
                    n_out_channels=n_outchannels,
                    dropout_rate=self.config.dropout_rate,
                    kernel_size=self.config.kernel_size,
                    n_groups=self.config.n_groups,
                )(hidden, is_training)
                if hidden.shape[-1] in self.config.attention_resolutions:
                    hidden = _AttentionBlock(n_heads=self.config.n_attention_heads, n_groups=self.config.n_groups)(
                        hidden, is_training
                    )

        outputs = nn.Conv(
            self.config.n_out_channels,
            kernel_size=(self.config.kernel_size, self.config.kernel_size),
            strides=(1, 1),
            padding="SAME",
            kernel_init=nn.initializers.zeros,
        )(hidden)

        return outputs


class VectorQuantizer(nn.Module):
    """Adopted from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/nets/vqvae.py#L41"""

    config: VectorQuantizerConfig

    def setup(self):
        self.embeddings = self.param(
            "embedding",
            nn.initializers.variance_scaling(1.0, "fan_in", distribution="uniform"),
            (self.config.embedding_dim, self.config.n_embedding),
        )

    def __call__(self, inputs, context, **kwargs):
        flat_inputs = jnp.reshape(inputs, [-1, self.config.embedding_dim])

        distances = (
            jnp.sum(jnp.square(flat_inputs), 1, keepdims=True)
            - 2 * jnp.matmul(flat_inputs, self.embeddings)
            + jnp.sum(jnp.square(self.embeddings), 0, keepdims=True)
        )

        encoding_indices = jnp.argmax(-distances, 1)
        encodings = jax.nn.one_hot(encoding_indices, self.config.n_embedding, dtype=distances.dtype)

        encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])
        quantized = self.quantize(encoding_indices)

        e_latent_loss = jnp.mean(jnp.square(jax.lax.stop_gradient(quantized) - inputs))
        q_latent_loss = jnp.mean(jnp.square(quantized - jax.lax.stop_gradient(inputs)))
        loss = q_latent_loss + self.config.commitment_cost * e_latent_loss

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
        self._encoders = [Encoder(self.encoder_configs[i], name=f"encoder_{i}") for i in range(self.n_scaling)]
        self._decoders = [
            Decoder(self.decoder_configs[self.n_scaling - 1 - i], name=f"decoder_{i}") for i in range(self.n_scaling)
        ]
        self._quantizers = [VectorQuantizer(self.vq_configs[i], name=f"quantizer_{i}") for i in range(self.n_scaling)]

    @nn.compact
    def __call__(self, inputs, is_training):
        encs = self._encode(inputs, is_training)
        decs, codes, quants = [], [], []
        vq_loss = 0
        for idx in list(reversed(range(self.n_scaling))):
            if not len(decs):
                vq_res = self._quantizers[idx](encs[idx], None)
                decoded = self._decoders[idx](vq_res["quantized"], is_training)
            else:
                appended = nn.Conv(encs[idx].shape[-1], kernel_size=(3, 3), strides=(1, 1), padding="SAME")(
                    jnp.concatenate([encs[idx], decs[-1]], axis=-1)
                )
                vq_res = self._quantizers[idx](appended, None)
                quantized = vq_res["quantized"]
                quantized_rest = self._upsample(quants, quantized.shape[1:3])
                quantized_all = jnp.concatenate([quantized, *quantized_rest], axis=-1)
                decoded = self._decoders[idx](quantized_all, is_training)
            vq_loss += vq_res["loss"]
            codes.append(vq_res["encodings"])
            quants.append(vq_res["quantized"])
            decs.append(decoded)

        outputs = decs[-1]
        recon_loss = jnp.mean(jnp.square(inputs - outputs))
        loss = recon_loss + vq_loss
        return {"outputs": outputs, "loss": loss}

    def _encode(self, inputs, is_training):
        encs = [inputs]
        for i in range(self.n_scaling):
            encoded = self._encoders[i](encs[-1], is_training=is_training)
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
            B, *_, C = quantized.shape
            upsampled = jax.image.resize(quantized, (B, *shape, C), method="nearest")
            upsampleds.append(upsampled)
        return upsampleds

    def decode(self, encodings, is_training):
        input_to_last_decoder = encodings[-1]
        sh = input_to_last_decoder.shape
        for i, input in enumerate(encodings[:-1]):
            upsampled = jax.image.resize(input, sh, method="nearest")
            input_to_last_decoder = jnp.concatenate([input_to_last_decoder, upsampled], axis=-1)
        output = self._decoders[0](input_to_last_decoder, is_training=is_training)
        return output

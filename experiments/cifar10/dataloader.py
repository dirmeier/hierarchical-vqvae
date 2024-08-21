import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import random as jr


def data_loaders(rng_key, config, split="train", outpath: str = None):
    datasets = tfds.load(
        "cifar10", try_gcs=False, split=split, data_dir=outpath, batch_size=-1
    )
    if isinstance(split, str):
        datasets = [datasets]
    itrs = []
    for dataset in datasets:
        ds = np.float32(dataset["image"]) / 255.0 - 0.5
        itr_key, rng_key = jr.split(rng_key)
        itr = _as_batched_numpy_iter(itr_key, ds, config)
        itrs.append(itr)
    return itrs


def _as_batched_numpy_iter(rng_key, itr, config):
    itr = tf.data.Dataset.from_tensor_slices(itr)
    max_int32 = jnp.iinfo(jnp.int32).max
    seed = jr.randint(rng_key, shape=(), minval=0, maxval=max_int32)
    return tfds.as_numpy(
        itr.shuffle(
            config.buffer_size,
            reshuffle_each_iteration=config.do_reshuffle,
            seed=int(seed),
        )
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(config.batch_size * 5)
    )

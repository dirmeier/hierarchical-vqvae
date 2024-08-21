import os
import pickle

import optax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint.utils import get_save_directory
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def new_train_state(rng_key, model, init_batch, config):
    variables = model.init(
        {"params": rng_key},
        inputs=init_batch,
        is_training=False,
    )
    if config.params.do_warmup and config.params.do_decay:
        lr = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.params.learning_rate,
            warmup_steps=config.params.warmup_steps,
            decay_steps=config.params.decay_steps,
            end_value=config.params.end_learning_rate,
        )
    elif config.params.do_warmup:
        lr = optax.linear_schedule(
            init_value=0.0,
            end_value=config.params.learning_rate,
            transition_steps=config.params.warmup_steps,
        )
    elif config.params.do_decay:
        lr = optax.cosine_decay_schedule(
            init_value=config.params.learning_rate,
            decay_steps=config.params.decay_steps,
            alpha=config.params.end_learning_rate / config.params.learning_rate,
        )
    else:
        lr = config.params.learning_rate

    tx = optax.adamw(lr, weight_decay=config.params.weight_decay)
    if config.params.do_gradient_clipping:
        tx = optax.chain(
            optax.clip_by_global_norm(config.params.gradient_clipping), tx
        )

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def save_pickle(outfile, obj):
    with open(outfile, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_checkpointer_fns(outfolder, config, model_config):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        save_interval_steps=config.save_interval_steps,
        create=True,
        best_fn=lambda x: x["val_loss"],
        best_mode="min",
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        outfolder,
        checkpointer,
        options,
    )
    save_pickle(os.path.join(outfolder, "config.pkl"), model_config)

    def save_fn(epoch, ckpt, metrics):
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(
            epoch, ckpt, save_kwargs={"save_args": save_args}, metrics=metrics
        )

    def restore_fn():
        return checkpoint_manager.restore(checkpoint_manager.best_step())

    def path_best_ckpt_fn():
        return get_save_directory(
            checkpoint_manager.best_step(), checkpoint_manager.directory
        )

    return save_fn, restore_fn, path_best_ckpt_fn

import hashlib
import os

import jax
import matplotlib.pyplot as plt
import wandb
from absl import app, flags, logging
from checkpointer import get_checkpointer_fns, new_train_state
from dataloader import data_loaders
from flax.training.early_stopping import EarlyStopping
from jax import random as jr
from ml_collections import config_flags

from vqvae import VQVAE, VAEConfig, VectorQuantizerConfig

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "model configuration")
flags.DEFINE_string("workdir", None, "work directory")
flags.DEFINE_bool("usewand", False, "use wandb for logging")
flags.mark_flags_as_required(["workdir", "config"])


def get_model(config):
    model = VQVAE(
        encoder_configs=[VAEConfig(**conf) for conf in config.encoder_configs],
        decoder_configs=[VAEConfig(**conf) for conf in config.decoder_configs],
        vq_configs=[VectorQuantizerConfig(**conf) for conf in config.vq_configs],
    )
    return model


@jax.jit
def step_fn(step_key, state, batch):
    def loss_fn(params, rngs):
        ret = state.apply_fn(
            variables={"params": params},
            rngs=rngs,
            inputs=batch,
            is_training=True,
        )
        return ret["loss"]

    loss, grads = jax.value_and_grad(loss_fn)(state.params, step_key)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def train_epoch(rng_key, state, train_iter):
    epoch_loss = 0.0
    for idx, batch in enumerate(train_iter):
        batch_key, rng_key = jr.split(rng_key)
        batch_loss, state = step_fn(batch_key, state, batch)
        epoch_loss += batch_loss
    epoch_loss /= idx
    return float(epoch_loss), state


def evaluate_model(rng_key, state, val_iter):
    @jax.jit
    def loss_fn(batch):
        ret = state.apply_fn(
            variables={"params": state.params},
            inputs=batch,
            is_training=False,
        )
        return ret["loss"]

    loss = 0.0
    for idx, batch in enumerate(val_iter):
        loss += loss_fn(batch)
    loss /= idx
    return float(loss)


def train(rng_key, model, config, train_iter, val_iter, model_id):
    logging.info("get train state and checkpointer")
    state_key, rng_key = jr.split(rng_key)
    state = new_train_state(state_key, model, next(iter(train_iter)), config.optimizer)
    ckpt_save_fn, ckpt_restore_fn, _ = get_checkpointer_fns(
        os.path.join(FLAGS.workdir, "checkpoints", model_id), config.training.checkpoints, config.model.to_dict()
    )

    logging.info("training model")
    early_stop = EarlyStopping(
        patience=config.training.early_stopping.n_patience, min_delta=config.training.early_stopping.min_delta
    )
    epoch_key, rng_key = jr.split(rng_key)
    for epoch in range(1, config.training.n_epochs + 1):
        train_key, val_key, sample_key = jr.split(jr.fold_in(epoch_key, epoch), 3)
        train_loss, state = train_epoch(train_key, state, train_iter)
        val_loss = evaluate_model(val_key, state, val_iter)
        logging.info(f"loss at epoch {epoch}: {train_loss}/{val_loss}")
        ckpt_save_fn(
            epoch,
            state,
            {"train_loss": train_loss, "val_loss": val_loss},
        )
        early_stop.update(val_loss)
        if FLAGS.usewand:
            wandb.log({"loss": train_loss, "val_loss": val_loss})
        # if FLAGS.usewand and epoch % 20 == 0:
        log_images(sample_key, state, val_iter, epoch, model_id)
        if early_stop.should_stop:
            logging.info("early stopping criterion found. stopping training")
            break


@jax.jit
def reconstruct(rng_key, state, batch):
    ret = state.apply_fn(
        variables={"params": state.params},
        inputs=batch,
        is_training=False,
    )
    return ret["outputs"]


def plot_figures(real_data, reconstructed_data):
    def convert_batch_to_image_grid(image_batch):
        reshaped = image_batch.reshape(4, 8, 32, 32, 3).transpose([0, 2, 1, 3, 4]).reshape(4 * 32, 8 * 32, 3)
        return reshaped + 0.5

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(convert_batch_to_image_grid(real_data), interpolation="nearest")
    ax.set_title("validation data originals")
    plt.axis("off")
    plt.tight_layout()
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(convert_batch_to_image_grid(reconstructed_data), interpolation="nearest")
    ax.set_title("validation data reconstructions")
    plt.axis("off")
    plt.tight_layout()
    return fig


def log_images(rng_key, state, val_iter, step, model_id):
    batch = next(iter(val_iter))
    outputs_hat = reconstruct(rng_key, state, batch)
    fig = plot_figures(batch, outputs_hat)
    if FLAGS.usewand:
        wandb.log({"images": wandb.Image(fig)}, step=step)
    fl = os.path.join(FLAGS.workdir, "figures", f"{model_id}-reconstructed_images-{step}.png")
    fig.savefig(fl)


def hash_value(config):
    h = hashlib.new("sha256")
    h.update(str(config).encode("utf-8"))
    return h.hexdigest()


def main(argv):
    del argv
    logging.set_verbosity(logging.INFO)
    config = FLAGS.config.to_dict()
    model_id = f"{hash_value(config)}"

    if FLAGS.usewand:
        wandb.init(
            project="vqvae-experiment",
            config=config,
            dir=os.path.join(FLAGS.workdir, "wandb"),
        )
        wandb.run.name = model_id

    rng_key = jr.PRNGKey(FLAGS.config.rng_key)
    data_key, train_key, rng_key = jr.split(rng_key, 3)
    train_iter, val_iter = data_loaders(
        rng_key=data_key,
        config=FLAGS.config.training,
        split=["train[:90%]", "train[90%:]"],
        outpath=os.path.join(FLAGS.workdir, "data"),
    )

    model = get_model(FLAGS.config.model)
    train(train_key, model, FLAGS.config, train_iter, val_iter, model_id)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)

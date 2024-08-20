import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 1
    config.model = new_dict(
        encoder_configs=[new_dict(n_channels=32, n_out_channels=32, channel_multipliers=(1, 2), n_resnet_blocks=2)] * 3,
        decoder_configs=[
            new_dict(n_channels=32, n_out_channels=32, channel_multipliers=(1, 2), n_resnet_blocks=2),
            new_dict(n_channels=32, n_out_channels=32, channel_multipliers=(1, 2), n_resnet_blocks=2),
            new_dict(n_channels=32, n_out_channels=3, channel_multipliers=(1, 2), n_resnet_blocks=2),
        ],
        vq_configs=[new_dict(n_embedding=32, embedding_dim=32, commitment_cost=0.1)] * 3,
    )

    config.training = new_dict(
        n_epochs=1,
        batch_size=32,
        buffer_size=64 * 5,
        prefetch_size=128,
        do_reshuffle=True,
        early_stopping=new_dict(n_patience=20, min_delta=0.001),
        checkpoints=new_dict(
            max_to_keep=5,
            save_interval_steps=5,
        ),
    )

    config.optimizer = new_dict(
        name="adamw",
        params=new_dict(
            learning_rate=1e-3,
            weight_decay=1e-4,
            do_warmup=False,
            warmup_steps=200_000,
            do_decay=True,
            decay_steps=500_000,
            end_learning_rate=1e-5,
            do_gradient_clipping=True,
            gradient_clipping=1.0,
        ),
    )

    return config

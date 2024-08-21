import ml_collections


def new_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.rng_key = 1
    config.model = new_dict(
        encoder_configs=[
            new_dict(
                n_channels=128,
                n_out_channels=128,
                n_layers=2,
                model_block_name="convnext",
            )
        ]
        * 3,
        decoder_configs=[
            new_dict(
                n_channels=128,
                n_out_channels=128,
                n_layers=2,
                model_block_name="convnext",
            ),
            new_dict(
                n_channels=128,
                n_out_channels=128,
                n_layers=2,
                model_block_name="convnext",
            ),
            new_dict(
                n_channels=128,
                n_out_channels=3,
                n_layers=2,
                model_block_name="convnext",
            ),
        ],
        vq_configs=[
            new_dict(embedding_dim=64, n_embedding=512, commitment_cost=0.25)
        ]
        * 3,
    )

    config.training = new_dict(
        n_epochs=50,
        batch_size=32,
        buffer_size=32 * 100,
        prefetch_size=32 * 4,
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
            weight_decay=1e-6,
            do_warmup=True,
            warmup_steps=2_000,
            do_decay=True,
            decay_steps=100_000,
            end_learning_rate=1e-5,
            do_gradient_clipping=True,
            gradient_clipping=1.0,
        ),
    )

    return config

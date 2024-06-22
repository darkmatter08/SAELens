# pip install --upgrade torch numpy ipython ipdb
# pip install -r requirements.txt
# pip install sae-lens transformer-lens circuitsvis

import argparse
import os

import torch

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

os.environ["TOKENIZERS_PARALLELISM"] = "false"


LEO_ACTIVATION_FN = "topk-32"

EXPERIMENTS = {
    "tiny": ("tiny-stories-1L-21M", 1024, LEO_ACTIVATION_FN),
    "gpt2-small": ("gpt2-small", 768, LEO_ACTIVATION_FN),
    "gpt2-xl": ("gpt2-xl", 1600, LEO_ACTIVATION_FN),
}
EXPERIMENTS_HYPERS = {
    "tiny": {"lr": 5e-5, "l1_coefficient": 5, "train_batch_size_tokens": 4096},
    "gpt2-small": {"lr": 1.20e-03, "l1_coefficient": 3, "train_batch_size_tokens": 4096},  # taken from `pretrained_saes.yaml`
    "gpt2-xl": {"lr": 1.20e-03, "l1_coefficient": 3, "train_batch_size_tokens": 32},  # taken from `pretrained_saes.yaml`
}

def configure_and_run(experiment: str = "gpt2-small", device: str = "cpu"):
    total_training_steps = 30_000  # probably we should do more
    batch_size = 4096
    batch_size = 32
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    model_to_use, d_in, activation_fn = EXPERIMENTS[experiment]

    lr = EXPERIMENTS_HYPERS[experiment]["lr"]
    l1_coefficient = EXPERIMENTS_HYPERS[experiment]["l1_coefficient"]
    train_batch_size_tokens = EXPERIMENTS_HYPERS[experiment]["train_batch_size_tokens"]
    total_training_tokens = total_training_steps * train_batch_size_tokens

    # CONTEXT_SIZE = 256
    CONTEXT_SIZE = 512

    cfg = LanguageModelSAERunnerConfig(
        ## Model Args
        model_name=model_to_use,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_name="blocks.0.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_layer=0,  # Only one layer in the model.
        d_in=d_in,  # the width of the mlp output.

        ## Dataset Args
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.

        ## SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",

        ## Training Parameters (for the SAE, the rest of the model is frozen)
        lr=lr,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=l1_coefficient,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=train_batch_size_tokens,
        context_size=CONTEXT_SIZE,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.

        ## Activation Store Parameters
        n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=16,

        ## Resampling protocol
        use_ghost_grads=False,  # we don't use ghost grads anymore.
        feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.

        ## WANDB
        log_to_wandb=False,  # always use wandb unless you are just testing code.
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,

        ## Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
        activation_fn=activation_fn,
    )

    # print all the important config parameters we overrode
    print(f"Running {experiment=}\nUsing {cfg.model_name=} {cfg.lr=} {cfg.l1_coefficient=} {cfg.train_batch_size_tokens=} {cfg.training_tokens=} {cfg.activation_fn=} {cfg.device=}")

    sparse_autoencoder = SAETrainingRunner(cfg).run()
    return sparse_autoencoder

if __name__ == "__main__":
    # import argparse and setup a parser to grab the experiment name
    # also add an option to override the device
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="gpt2-small")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not args.device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print("Using device:", device)
    configure_and_run(experiment=args.experiment, device=device)

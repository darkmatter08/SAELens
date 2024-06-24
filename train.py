# pip install -r requirements.txt
# pip install --upgrade torch numpy ipython ipdb sae-lens transformer-lens circuitsvis

import argparse
import os
import subprocess

import torch

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if "NO_WANDB" in os.environ:
    wandb = False
else:
    wandb = True

def get_git_commit_hash():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting the commit hash: {e}")
        return None


# These were selected based on gpt2-xl using sweep_{lr,bs}.sh
BEST_LR = 1.20e-04  # note that this was the best lr for gpt2-xl; gpt2-small hasn't been swept, but SAELens used `1.2e-03`.
BEST_BATCH_SIZE = 512  # note that this was the largest bs tested with gpt2-xl: (64 128 256 512); there may be larger ones that work.
# on gpt2-xl:
# 1024 works, but is slow (2hr train), and reduces gpu util
# 2048 works, but is slow (4hr train), and reduces gpu util
# 4096 results in CUDA OOM
# Based on this dramatic slowdown, I am going to use 512 for gpt2-xl

EXPERIMENTS = {
    # Disable l1 regularization for models using topk
    "tiny": {
        "model_name": "tiny-stories-1L-21M", "d_in": 1024, "activation_fn": "topk-32",
        "lr": 5e-5, "l1_coefficient": 0, "train_batch_size_tokens": 4096, "context_size": 512,
        "total_training_steps": 30_000,
    },
    # Note: 4096 works for gpt2-small, but for consistency, I am using 512 here.
    "gpt2-small": {
        "model_name": "gpt2-small", "d_in": 768, "activation_fn": "topk-32",
        "lr": 1.20e-03, "l1_coefficient": 0, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 512,
        "total_training_steps": 30_000,
    },
    "gpt2-small-debug": {
        "model_name": "gpt2-small", "d_in": 768, "activation_fn": "topk-32",
        "lr": 1.20e-03, "l1_coefficient": 0, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 512,
        "total_training_steps": 30_000,
    },
    "gpt2-small-relu": {
        "model_name": "gpt2-small", "d_in": 768, "activation_fn": "relu",
        "lr": 1.20e-03, "l1_coefficient": 1.80, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 512,
        "total_training_steps": 30_000,
    },
    # Note: we shrunk the LR to improve the learning curve for gpt2-xl; however this may not actually work.
    "gpt2-xl": {
        "model_name": "gpt2-xl", "d_in": 1600, "activation_fn": "topk-32",
        "lr": BEST_LR, "l1_coefficient": 0, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 256,
        "total_training_steps": 30_000,
    },
    "gpt2-xl-debug": {
        "model_name": "gpt2-xl", "d_in": 1600, "activation_fn": "topk-32",
        "lr": BEST_LR, "l1_coefficient": 0, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 256,
        "total_training_steps": 30_000,
    },
    "gpt2-xl-relu": {
        "model_name": "gpt2-xl", "d_in": 1600, "activation_fn": "relu",
        "lr": BEST_LR, "l1_coefficient": 1.80, "train_batch_size_tokens": BEST_BATCH_SIZE, "context_size": 256,
        "total_training_steps": 30_000,
    },
}

def configure_and_run(experiment_str: str = "gpt2-small", device: str = "cpu", hyper_overrides: dict[str, int | float] = {}):
    experiment = EXPERIMENTS[experiment_str]

    for hyper, value in hyper_overrides.items():
        if hyper in experiment:
            experiment[hyper] = value
        else:
            raise ValueError(f"attemtped to override hyper that is not set: {hyper=} {value=}")

    model_name, d_in, activation_fn = experiment["model_name"], experiment["d_in"], experiment["activation_fn"]

    lr, l1_coefficient, train_batch_size_tokens, context_size = experiment["lr"], experiment["l1_coefficient"], experiment["train_batch_size_tokens"], experiment["context_size"]

    total_training_steps: int = experiment["total_training_steps"]  # we should probably do >30k

    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    total_training_tokens = total_training_steps * train_batch_size_tokens

    cfg = LanguageModelSAERunnerConfig(
        ## Model Args
        model_name=model_name,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
        hook_name="blocks.8.hook_mlp_out",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_layer=8,  # Only one layer in the model.
        d_in=d_in,  # the width of the mlp output.

        ## Dataset Args
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        # dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
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
        # normalize_activations="expected_average_only_in",
        normalize_activations="none",

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
        context_size=context_size,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.

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
        log_to_wandb=wandb,  # always use wandb unless you are just testing code.
        wandb_project="sae_lens_sweeps",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,

        ## Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
        activation_fn=activation_fn,
        aux_k_coefficient=1.0/32.0,
    )

    print(f"Running {experiment=}\nUsing {cfg.device=}")

    sparse_autoencoder = SAETrainingRunner(cfg).run()
    return sparse_autoencoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="gpt2-small", choices=EXPERIMENTS.keys())
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=str, default=None)
    parser.add_argument("--bs", type=int, default=None)
    args = parser.parse_args()

    print(f"commit: {get_git_commit_hash()}")
    if not args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    hyper_overrides = {}
    if args.lr:
        try:
            float(args.lr)
        except ValueError:
            raise ValueError(f"lr must be a float, got {args.lr}")
        print(f"overriding lr with {float(args.lr)}")
        hyper_overrides.update({"lr": float(args.lr)})
    elif args.bs:
        print(f"overriding bs with {args.bs}")
        hyper_overrides.update({"train_batch_size_tokens": args.bs})
    else:
        configure_and_run(experiment_str=args.experiment, device=args.device, hyper_overrides=hyper_overrides)


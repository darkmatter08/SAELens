#!/bin/bash

# Get the model size from the command line
model_size=$1

if [[ "$model_size" != "xl" && "$model_size" != "small" ]]; then
    echo "Usage: $0 {xl|small}"
    exit 1
fi

if [[ "$model_size" == "xl" ]]; then
    experiments=(gpt2-xl gpt2-xl-relu)
else
    experiments=(gpt2-small gpt2-small-relu)
fi
for experiment in "${experiments[@]}"; do
    python train.py --experiment $experiment
done

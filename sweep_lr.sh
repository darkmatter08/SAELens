lr_values=(0.0006000000000000001 0.00012 1.2e-05 1.2e-06)

for lr in "${lr_values[@]}"; do
    python train.py --experiment gpt2-xl --lr $lr
done

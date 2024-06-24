bs_values=(64 128 256 512)

for bs in "${bs_values[@]}"; do
    python train.py --experiment gpt2-xl --bs $bs
done

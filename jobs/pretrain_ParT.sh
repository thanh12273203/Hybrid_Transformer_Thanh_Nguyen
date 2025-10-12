#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --cpus-per-task 128
#SBATCH --image=docker:thanh12273203/gsoc25_cms:latest
#SBATCH --output=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --mail-user=tpnguyen8@crimson.ua.edu
#SBATCH --mail-type=ALL

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

srun --unbuffered --export=ALL shifter python -m scripts.train_ParT \
    --config-path ./configs/pretrain_ParT.yaml \
    --train-data-dir ./data/train_100M \
    --val-data-dir ./data/val_5M

# Remember to change --best-model-path to the path of the best model you want to evaluate
srun --unbuffered --export=ALL shifter python -m scripts.evaluate_ParT \
    --config-path ./configs/pretrain_ParT.yaml \
    --best-model-path ./logs/ParticleTransformer/best/run_01.pt \
    --test-data-dir ./data/test_20M
#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH -q debug
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --cpus-per-task 64
#SBATCH --image=docker:thanh12273203/gsoc25_cms:latest
#SBATCH --output=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --mail-user=tpnguyen8@crimson.ua.edu
#SBATCH --mail-type=ALL

nvidia-smi
export CUDA_VISIBLE_DEVICES=0, 1, 2, 3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
srun --unbuffered --export=ALL shifter python -m scripts.train
#!/bin/bash
#SBATCH --job-name=mtt_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # Number of processes (GPUs) per node
#SBATCH --gres=gpu:4                  # Adjust if you have more GPUs
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=4g
#SBATCH --time=04:00:00

module load anaconda
module load cuda
conda activate xrayai

# Launch distributed training using torchrun
torchrun \
  --nproc_per_node=2 \
  --master_port=12355 \
  train_ddp.py

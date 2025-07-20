#!/bin/bash
#SBATCH --job-name=mtt-train
#SBATCH --output=logs/train_%j.log
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4

module load cuda/12.1 anaconda
source activate xrayai

python train_mtt.py --num_samples 1 --batch_size 1 --epochs 60 --output_dir results/run_%j

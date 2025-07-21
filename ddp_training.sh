#!/bin/bash -l
#SBATCH -J cnn-mtt               # job name
#SBATCH --time=00-00:60:00           # requested time (DD-HH:MM:SS)
#SBATCH -p gpu,preempt               # partitions to run on
#SBATCH -N 1                        # number of nodes
#SBATCH -n 4                        # number of tasks total (processes)
#SBATCH --mem=96g                    # total RAM requested (increase if needed)
#SBATCH --gres=gpu:4           # request 4 A100 GPUs
#SBATCH --output=cnn-mtt.%j.%N.out    # stdout file
#SBATCH --error=cnn-mtt.%j.%N.err     # stderr file
#SBATCH --mail-type=ALL             # email notifications on all events
#SBATCH --mail-user=dbanco02@tufts.edu

module purge

module load anaconda
module load cuda
source activate xrayai

export NCCL_P2P_LEVEL=NVL

# Launch distributed training using torchrun with 4 GPUs
torchrun --nproc_per_node=4 --master_port=12355 train_ddp.py

conda deactivate
#!/bin/bash
#SBATCH --job-name=mtt-gen
#SBATCH --output=mtt-gen-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

module load anaconda
conda activate xrayai

python saveMTTdataset.py --out mtt_dataset_30000x30x32x96.pt --N 30000 --T 30 --H 32 --W 96 --spots 3 --noise

#!/bin/bash
#SBATCH --job-name=Embedding_FindQuote
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=1:00:00
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

module load anaconda3
# Activate existing conda environment
source activate cs3600-llm

python -u find_quote.py

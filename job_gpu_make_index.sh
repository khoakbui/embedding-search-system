#!/bin/bash
#SBATCH --job-name=Embedding_MakeIndex
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=2:00:00
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

module load anaconda3

# Create or activate conda environment
if conda env list | grep -q "cs3600-llm"; then
    echo "Using existing conda environment..."
else
    echo "Creating conda environment..."
    conda env create -f environment.yml
fi

source activate cs3600-llm

python -u make_index.py

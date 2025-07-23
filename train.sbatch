#!/bin/bash
#SBATCH -C m16
#SBATCH -c 45
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 5G
#SBATCH --time 12:00:00

source venv/bin/activate

export PYTHONPATH=$(pwd):$PYTHONPATH
srun python terratorch_co2.py fit --config configs/ben-s1.yaml
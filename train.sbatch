#!/bin/bash
#SBATCH -C m16
#SBATCH -c 5
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 11G
#SBATCH --time 6-12:00:00
#SBATCH --partition longrun

source venv/bin/activate

export PYTHONPATH=$(pwd):$PYTHONPATH
srun python terratorch_co2.py fit --config configs/base/segmentation.yaml --config configs/sen12ms-s2.yaml
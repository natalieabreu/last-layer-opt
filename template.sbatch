#!/bin/bash
#SBATCH -J last-layer-opt
#SBATCH --array=0
#SBATCH -o printouts/gn/%A_%a.out
#SBATCH -e printouts/gn/%A_%a.err
#SBATCH -p kempner_h100
#SBATCH --account=
#SBATCH --constraint="h100"
#SBATCH -t 0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1000G
#SBATCH -c 24

module load python/3.10.12-fasrc01;
mamba activate opt;
export PYTHONPATH="${PYTHONPATH}:/"
export WANDB_API_KEY=

python gradient_norms.py

mamba deactivate
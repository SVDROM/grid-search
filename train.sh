#!/bin/sh
#SBATCH --qos turing
#SBATCH --account vjgo8416-dmd-ddwm
#SBATCH --time 8:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-gpu 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem 128G
#SBATCH --job-name dmd_grid_search
#SBATCH --output dmd_grid_search.log

source ../load_python_3_11.sh

source ../.venv/bin/activate
echo $(which python)

python train.py

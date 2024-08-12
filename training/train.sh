#!/bin/sh

#SBATCH --job-name=dashi2_a100
#SBATCH --error=output.err
#SBATCH --output=output.log
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# conda
. "/userspace/cdd/miniconda3/etc/profile.d/conda.sh"
conda activate pycuda

export HF_DATASETS_CACHE="/userspace/cdd/cash"
export HF_HOME="/userspace/cdd/cash"
export MPLCONFIGDIR="/userspace/cdd/cash"

export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}

# exec
python -u train.py
python -V

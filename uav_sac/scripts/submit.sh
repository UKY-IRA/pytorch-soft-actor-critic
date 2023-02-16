#!/bin/bash

#SBATCH --partition=V4V32_SKY32M192_L
#SBATCH --cpus-per-task=4
#SBATCH -A gol_bxi224_uksr
#SBATCH --job-name=train_qdn
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=out.out
#SBATCH -e slurm-%j.err  # Error file for this job.
#SBATCH -o slurm-%j.out  # Output file for this job.

module purge
module load ccs/cuda/11.2.0_460.27.04
module load ccs/singularity
singularity shell /share/singularity/images/ccs/conda/lcc-conda-1-centos8.sinf
. activate $PSCRATCH/bxi224_uksr/pytorch-soft-actor-critic/env
export PYTHONUNBUFFERED=TRUE # slow but gives you sysout back
python $PSCRATCH/bxi224_uksr/pytorch-soft-actor-critic/uav_sac/main.py \
$PSCRATCH/bxi224_uksr/pytorch-soft-actor-critic/uav_sac/training_config.json
conda deactivate

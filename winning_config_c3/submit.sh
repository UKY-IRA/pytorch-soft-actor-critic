#!/bin/bash

#SBATCH --partition=V4V32_SKY32M192_L
#SBATCH --cpus-per-task=4
#SBATCH -A gol_bxi224_uksr
#SBATCH --job-name=train_qdn
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=out.out
#SBATCH -e slurm-%j.err  # Error file for this job.
#SBATCH -o slurm-%j.out  # Output file for this job.

module purge
module load ccs/cuda/10.0.130
module load ccs/singularity
singularity shell /share/singularity/images/ccs/conda/lcc-conda-1-centos8.sinf
. activate $PSCRATCH/bxi224_uksr/plane_sac/env
export PYTHONUNBUFFERED=TRUE # slow but gives you sysout back
python $PSCRATCH/bxi224_uksr/plane_sac/main.py \
  --env-name Plane \
  --policy Gaussian \
  --eval 100 \
  --gamma 0.9 \
  --tau 0.007 \
  --lr 0.0001 \
  --alpha 0.1 \
  --automatic_entropy_tuning True \
  --hidden_size 256 \
  --batch_size 512 \
  --seed 1281341 \
  --target_update_interval 20 \
  --start_steps 10000 \
  --replay_size 100000 \
  --num_steps 2000001 \
  --updates_per_step 0.333 \
  --cuda
conda deactivate

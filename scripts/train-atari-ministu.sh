#!/bin/bash
#SBATCH --job-name=ezv2-atari-ministu
#SBATCH --mail-user=cb4835@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=output-ministu-%j.out
#SBATCH --error=error-ministu-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00           
#SBATCH --gres=gpu:1             
#SBATCH --constraint="nomig&gpu40"  

conda activate ezv2-flash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# MiniSTU-based EfficientZero with sequence_length=20
python ~/EfficientZero-v2/ez/train.py \
  exp_config=~/EfficientZero-v2/ez/config/exp/atari.yaml \
  model.use_mini_stu_dynamics=True \
  model.mini_stu.sequence_length=20

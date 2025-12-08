#!/bin/bash
#SBATCH --job-name=ezv2-atari-classical
#SBATCH --mail-user=cb4835@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output=output-classical-%j.out
#SBATCH --error=error-classical-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00           
#SBATCH --gres=gpu:1             
#SBATCH --constraint="nomig&gpu80"  

conda activate ezv2-flash

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# Classical EfficientZero (no MiniSTU)
python ~/EfficientZero-v2/ez/train.py exp_config=~/EfficientZero-v2/ez/config/exp/atari.yaml model.use_mini_stu_dynamics=False

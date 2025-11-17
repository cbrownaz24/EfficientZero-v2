#!/bin/bash
#SBATCH --job-name=ez_train
#SBATCH --gres=gpu:2
#SBATCH --constraint="nomig&gpu80"
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ez_%j.out
#SBATCH --error=logs/ez_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cb4835@princeton.edu

source ~/.bashrc

conda activate ezv2

cd /home/cb4835/EfficientZero-v2

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

mkdir -p logs

python ez/train.py exp_config=ez/config/exp/pong.yaml
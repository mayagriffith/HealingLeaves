#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=LAYOUT_CONTROLNET
#SBATCH --mail-user=mgriff13@uvm.edu
#SBATCH --mail-type=ALL

#training

python3 -u train.py \
    --batch_size 1 \
    --lr 1e-5 \
    --logger_freq 1 \
    --gpu 1 \
    --min_epochs 1 \
    --max_epochs 40 \
    --exp_name "f_BEV predicted segmentation with ground" \


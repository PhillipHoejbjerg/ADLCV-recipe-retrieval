#!/bin/sh
#BSUB -J pretrained_both
#BSUB -o /work3/s184984/repos/ADLCV-recipe-retrieval/sh_logs/pretrained_both.out
#BSUB -e /work3/s184984/repos/ADLCV-recipe-retrieval/sh_errors/pretrained_both.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 4:00
#BSUB -R 'rusage[mem=30GB]'
#BSUB -R 'span[hosts=1]'

nvidia-smi
module load cuda/11.8
source /work3/s184984/repos/ADLCV-recipe-retrieval/recipe/bin/activate

CUDA_VISIBLE_DEVICES=0 python src/models/train_model.py \
    --experiment_name pretrained_both\
    --num_epochs 100 \
    --loss_fn cosine \
    --p 0.5 \

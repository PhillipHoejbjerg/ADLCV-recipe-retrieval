#!/bin/sh
#BSUB -J contrastive
#BSUB -o /work3/s184984/repos/ADLCV-recipe-retrieval/sh_logs/contrastive.out
#BSUB -e /work3/s184984/repos/ADLCV-recipe-retrieval/sh_errors/contrastive.err
#BSUB -n 6
#BSUB -q gpua100
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -W 4:00
#BSUB -R 'rusage[mem=30GB]'
#BSUB -R 'span[hosts=1]'
#BSUB -B
#BSUB -N

nvidia-smi
module load cuda/11.8
source /work3/s184984/repos/ADLCV-recipe-retrieval/recipe/bin/activate

CUDA_VISIBLE_DEVICES=0 python src/models/train_model.py \
    --experiment_name contrastive \
    --num_epochs 100 \
    --loss_fn contrastive

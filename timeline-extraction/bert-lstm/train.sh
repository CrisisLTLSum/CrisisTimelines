#!/bin/bash
#SBATCH -p nvidia
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

python bert_lstm.py \
    --bert_model /scratch/ba63/BERT_models/bert-base-uncased \
    --train_file /scratch/ba63/CrisisLTLSum/data/train.json \
    --dev_file /scratch/ba63/CrisisLTLSum/data/dev.json \
    --num_epochs 50 \
    --learning_rate 5e-5 \
    --hidd_size 128 \
    --num_layers 1 \
    --batch_size 16 \
    --dropout 0.1 \
    --do_train \
    --seed 42

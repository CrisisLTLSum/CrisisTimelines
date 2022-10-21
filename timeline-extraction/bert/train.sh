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

python run_classifier.py \
    --model_name_or_path  /scratch/ba63/BERT_models/bert-base-uncased \
    --do_train \
    --train_file /scratch/ba63/CrisisLTLSum/data/train.json \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --relevant_tweets_only \
    --output_dir /scratch/ba63/CrisisLTLSum/bert-relevant-only \
    --per_device_train_batch_size 32 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir

# python run_classifier.py \
#     --model_name_or_path  /scratch/ba63/BERT_models/bert-base-uncased \
#     --do_train \
#     --train_file /scratch/ba63/CrisisLTLSum/data/train.json \
#     --num_train_epochs 10 \
#     --max_seq_length 512 \
#     --pad_to_max_length \
#     --output_dir /scratch/ba63/CrisisLTLSum/bert \
#     --per_device_train_batch_size 32 \
#     --seed 42 \
#     --overwrite_cache \
#     --overwrite_output_dir



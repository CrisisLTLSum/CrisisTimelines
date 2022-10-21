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

# facebook/bart-base
# sshleifer/distilbart-cnn-12-6

python run_summarization.py \
     --model_name_or_path /scratch/ba63/BERT_models/distilbart-cnn-12-6 \
     --do_train \
     --train_file  /scratch/ba63/CrisisLTLSum/data/train.sum.json \
     --num_train_epochs 10 \
     --text_column timeline \
     --summary_column summary \
     --output_dir /scratch/ba63/CrisisLTLSum/timeline-summarization/distilbart \
     --per_device_train_batch_size 16 \
     --max_target_length 512 \
     --seed 42 \
     --overwrite_output_dir \
     --overwrite_cache


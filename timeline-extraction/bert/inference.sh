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

python inference.py \
    --eval_data_file /scratch/ba63/CrisisLTLSum/data/test.json \
    --model_path  /scratch/ba63/CrisisLTLSum/timeline-extraction/bert-relevant-only/checkpoint-2000-best \
    --max_seq_length 512

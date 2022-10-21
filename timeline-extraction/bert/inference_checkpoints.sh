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

for checkpoint in /scratch/ba63/CrisisLTLSum/timeline-extraction/bert-relevant-only/checkpoint*
    do
        echo "Inference using $checkpoint..."
        python inference.py \
            --eval_data_file /scratch/ba63/CrisisLTLSum/data/dev.json \
            --model_path  $checkpoint \
            --max_seq_length 512
    done

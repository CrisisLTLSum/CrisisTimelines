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


# ORACEL:
# BEST BART: /scratch/ba63/CrisisLTLSum/timeline-summarization/bart
# BEST DISTILLBART: /scratch/ba63/CrisisLTLSum/timeline-summarization/distilbart/checkpoint-500

# Non-ORACLE:
# BEST BART: /scratch/ba63/CrisisLTLSum/timeline-summarization/bart
# BEST DISTILLBART: /scratch/ba63/CrisisLTLSum/timeline-summarization/distilbart/checkpoint-500

python run_summarization.py \
    --model_name_or_path /scratch/ba63/CrisisLTLSum/timeline-summarization/bart \
    --do_predict \
    --validation_file /scratch/ba63/CrisisLTLSum/data/test.sum.bert.relevant.json \
    --test_file  /scratch/ba63/CrisisLTLSum/data/test.sum.bert.relevant.json \
    --text_column timeline \
    --summary_column summary \
    --per_device_eval_batch_size 16 \
    --output_dir /scratch/ba63/CrisisLTLSum/timeline-summarization/bart \
    --num_beams 4 \
    --max_target_length 512 \
    --predict_with_generate \
    --output_file_name test.sum.bert.relevant.txt \
    --overwrite_cache

# for checkpoint in  /scratch/ba63/CrisisLTLSum/timeline-summarization/bart /scratch/ba63/CrisisLTLSum/timeline-summarization/bart/checkpoint*
# do
# echo "inference using $checkpoint"

#     python run_summarization.py \
#         --model_name_or_path $checkpoint \
#         --do_predict \
#         --validation_file /scratch/ba63/CrisisLTLSum/data/dev.sum.bert.relevant.json \
#         --test_file  /scratch/ba63/CrisisLTLSum/data/dev.sum.bert.relevant.json \
#         --text_column timeline \
#         --summary_column summary \
#         --per_device_eval_batch_size 16 \
#         --output_dir $checkpoint \
#         --num_beams 4 \
#         --max_target_length 512 \
#         --predict_with_generate \
#         --output_file_name dev.sum.bert.relevant.txt \
#         --overwrite_cache

# done

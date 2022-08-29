# python run_classifier.py \
#     --model_name_or_path  bert-base-uncased \
#     --do_train \
#     --train_file /home/balhafni/timeline-summarization/new_data/train.json \
#     --num_train_epochs 10 \
#     --max_seq_length 512 \
#     --pad_to_max_length \
#     --relevant_tweets_only \
#     --output_dir models_new/bert-relevant-only \
#     --per_device_train_batch_size 16 \
#     --seed 42 \
#     --overwrite_cache \
#     --overwrite_output_dir

python run_classifier.py \
    --model_name_or_path  bert-base-uncased \
    --do_train \
    --train_file /home/balhafni/timeline-summarization/new_data/train.json \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --output_dir models_new/bert \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir



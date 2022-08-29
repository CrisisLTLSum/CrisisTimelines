# facebook/bart-base
# sshleifer/distilbart-cnn-12-6

# python run_summarization.py \
#      --model_name_or_path facebook/bart-base \
#      --do_train \
#      --train_file  /home/balhafni/timeline-summarization/new_data/train.sum.json \
#      --num_train_epochs 10 \
#      --text_column timeline \
#      --summary_column summary \
#      --output_dir models_new/bart \
#      --per_device_train_batch_size 8 \
#      --max_target_length 512 \
#      --seed 42 \
#      --overwrite_output_dir \
#      --overwrite_cache


# python run_summarization.py \
#     --model_name_or_path sshleifer/distilbart-cnn-12-6 \
#     --do_train \
#     --train_file  /home/balhafni/timeline-summarization/new_data/train.sum.json \
#     --num_train_epochs 10 \
#     --text_column timeline \
#     --summary_column summary \
#     --output_dir models_new/distill-bart \
#     --per_device_train_batch_size 4 \
#     --max_target_length 512 \
#     --seed 42 \
#     --overwrite_output_dir \
#    --overwrite_cache 

python run_summarization.py \
     --model_name_or_path facebook/bart-base \
     --do_train \
     --train_file  /home/balhafni/timeline-summarization/data/extractive_models_outputs/sent_level_combs/sent_level_combs.oracle.train.json \
     --num_train_epochs 10 \
     --text_column timeline \
     --summary_column summary \
     --output_dir extractive_with_bart_models/sent_level_combs \
     --per_device_train_batch_size 8 \
     --max_target_length 512 \
     --seed 42 \
     --overwrite_output_dir \
     --overwrite_cache


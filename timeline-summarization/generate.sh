# python run_summarization.py \
#     --model_name_or_path sshleifer/distilbart-cnn-12-6 \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/new_data/dev.sum.bert.relevant.json \
#     --test_file  /home/balhafni/timeline-summarization/new_data/dev.sum.bert.relevant.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --output_dir models_new/zero-shot-distill-bart \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_file_name dev.sum.bert.relevant.txt \
#     --overwrite_cache

# for checkpoint in  models_new/bart models_new/bart/checkpoint*
# do
# echo "inference using $checkpoint"

# python run_summarization.py \
#     --model_name_or_path $checkpoint \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/new_data/dev.sum.bert.relevant.json \
#     --test_file  /home/balhafni/timeline-summarization/new_data/dev.sum.bert.relevant.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --output_dir $checkpoint \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_file_name dev.sum.bert.relevant.txt \
#     --overwrite_cache

# done

# models-new/bart/checkpoint-1000
# python run_summarization.py \
#     --model_name_or_path models-new/bart/checkpoint-1000 \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/data/test.sum.bert.relevant.json \
#     --test_file  /home/balhafni/timeline-summarization/data/test.sum.bert.relevant.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --output_dir models-new/bart/checkpoint-1000 \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_file_name test.sum.bert.relevant.txt \
#     --overwrite_cache


# Running inference on test.
# bart/ was the best on dev.oracle, distill bart checkpoint-1500-best-oracle was best on dev.oracle)
# bart/checkpoint-1000-best was the best on dev.sum.bert.relevant and distill-bart/checkpoint-500-best was best on dev.sum.bert.relevant

# python run_summarization.py \
#     --model_name_or_path models_new/bart/checkpoint-1000-best \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/new_data/test.sum.bert.relevant.json \
#     --test_file  /home/balhafni/timeline-summarization/new_data/test.sum.bert.relevant.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --output_dir models_new/bart/checkpoint-1000-best \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_file_name test.sum.bert.relevant.txt \
#     --overwrite_cache


# python run_summarization.py \
#     --model_name_or_path models_new/bart/ \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/data/extractive_models_outputs/tweet_level_combs.oracle.dev.json \
#     --test_file  /home/balhafni/timeline-summarization/data/extractive_models_outputs/tweet_level_combs.oracle.dev.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_dir models_new/bart/extractive_oracle_exp \
#     --output_file_name tweet_level_combs.oracle.dev.txt \
#     --overwrite_cache

# EXP=sent_level_combs
# python run_summarization.py \
#     --model_name_or_path extractive_with_bart_models/${EXP} \
#     --do_predict \
#     --validation_file /home/balhafni/timeline-summarization/data/extractive_models_outputs/${EXP}/${EXP}.oracle.dev.json \
#     --test_file  /home/balhafni/timeline-summarization/data/extractive_models_outputs/${EXP}/${EXP}.oracle.dev.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_dir extractive_with_bart_models/${EXP}/ \
#     --output_file_name ${EXP}.oracle.dev.txt \
#     --overwrite_cache

# python run_summarization.py \
#     --model_name_or_path extractive_with_bart_models/sent_level \
#     --do_predict \
#     --validation_file /home/balhafni/PreSumm/logs_dev_sent_level/sent_level.presum.dev.json \
#     --test_file  /home/balhafni/PreSumm/logs_dev_sent_level/sent_level.presum.dev.json \
#     --text_column timeline \
#     --summary_column summary \
#     --per_device_eval_batch_size 16 \
#     --num_beams 4 \
#     --max_target_length 512 \
#     --predict_with_generate \
#     --output_dir extractive_with_bart_models/sent_level/ \
#     --output_file_name sent_level.presum.dev.txt \
#     --overwrite_cache


python run_summarization.py \
    --model_name_or_path models_new/bart \
    --do_predict \
    --validation_file /home/balhafni/PreSumm/logs_dev_sent_level/sent_level.presum.dev.json \
    --test_file  /home/balhafni/PreSumm/logs_dev_sent_level/sent_level.presum.dev.json \
    --text_column timeline \
    --summary_column summary \
    --per_device_eval_batch_size 16 \
    --num_beams 4 \
    --max_target_length 512 \
    --predict_with_generate \
    --output_dir models_new/bart/extractive_presum_exp \
    --output_file_name sent_level.presum.dev.txt \
    --overwrite_cache
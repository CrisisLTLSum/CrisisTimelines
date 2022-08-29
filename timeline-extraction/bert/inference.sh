# models/bert-relevant-only/checkpoint-4500

python inference.py \
    --eval_data_file /home/balhafni/timeline-summarization/data/test.json \
    --model_path models_new/bert-relevant-only/checkpoint-4500-best \
    --max_seq_length 512

for checkpoint in models_new/bert-relevant-only models_new/bert-relevant-only/checkpoint*
do
echo "Inference using $checkpoint..."
python inference.py \
    --eval_data_file /home/balhafni/timeline-summarization/new_data/dev.json \
    --model_path  $checkpoint \
    --max_seq_length 512
done

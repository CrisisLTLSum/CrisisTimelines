# CrisisLTLSum: A Benchmark for Local Crisis Event Timeline Extraction and Summarization

This repo contains code and pretrained models to reproduce results in our paper [CrisisLTLSum: A Benchmark for Local Crisis Event Timeline Extraction and Summarization]().

## Requirements:
The code was written for python>=3.8, pytorch 1.11.0, and transformers 4.19.0. You will also need a few additional packages. Here's how you can set up the environment using conda:

```bash
git clone https://github.com/balhafni/CrisisLTLSum.git
cd CrisisLTLSum

conda create -n crisis_sum python=3.8
conda activate crisis_sum

pip install -r requirements.txt
```

## Experiments:

### Timeline Extraction:

Code to run experiments for the timeline extraction task can be found in [timeline-extraction](timeline-extraction). There are three types of experiments:

1) Majority class baseline: The results for this experiment can be simply obtained by running `python simple_baselines.py`.

2) BERT Sequence Classification: There are two modes to run this experiment. In the first mode, the context is simply constructed by concatenating all the tweets to create examples that are used to fine-tuned BERT. To run this experiment, you have to run:

```bash
python run_classifier.py \
    --model_name_or_path  bert-base-uncased \
    --do_train \
    --train_file data/train.json \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --output_dir models/bert \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
```

In the second mode, we create the context by only concatenating the relevant tweets only. To run this experiment, you have to run:

```bash
python run_classifier.py \
    --model_name_or_path  bert-base-uncased \
    --do_train \
    --train_file data/train.json \
    --num_train_epochs 10 \
    --max_seq_length 512 \
    --pad_to_max_length \
    --relevant_tweets_only \
    --output_dir models/bert-relevant-only \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --overwrite_cache \
    --overwrite_output_dir
```

For inference, take a look at [inference.sh](timeline-extraction/bert/inference.sh).

3) BERT-GRU Sequence Labeling: To run the BERT-GRU experiment, take a look at [train.sh](timeline-extraction/bert-lstm/train.sh) and for inference, take a look at [inference.sh](timeline-extraction/bert-lstm/inference.sh).


### Timeline Summarization:


## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more information.



## Citations:
If you find the code or data in this repo helpful, please cite [our paper]():

ADD BIBTEX

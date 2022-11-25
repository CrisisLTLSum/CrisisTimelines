# CrisisLTLSum: A Benchmark for Local Crisis Event Timeline Extraction and Summarization

This repo contains code and pretrained models to reproduce results in our paper [CrisisLTLSum: A Benchmark for Local Crisis Event Timeline Extraction and Summarization](https://arxiv.org/pdf/2210.14190.pdf).

## Requirements:
The code was written for python>=3.8, pytorch 1.11.0, and transformers 4.19.0. You will also need a few additional packages. Here's how you can set up the environment using conda:

```bash
git clone https://github.com/CrisisLTLSum/CrisisTimelines.git
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

#### Evaluation Results on the Test Set:


|              | Accuracy     | 
| -----------  | -----  | 
| **BERT**     |   73.51    |
| **Human**     |   **88.96**    |


### Timeline Summarization:
Code to run experiments for the timeline summarization task can be found in [timeline-summarization](timeline-summarization).

1) Naive baselines: To run the naive baseline experiments, you simply have to run `python simple_baselines.py`. This will output 3 files: `first_tweet_baseline.txt`, `last_tweet_baseline.txt`, and `random_tweet_baseline.txt` for each of the baselines, respectively. 

2) Fine-tuning BART/Distill-BART: To fine-tune BART or Distill-BART, take a look at [run_summarization.sh](timeline-summarization/run_summarization.sh)

3) Inference: The inference could be run in two modes: a) Oracle; b) After timeline extraction.

**Notes**: Since each timeline in our dataset is annotated by three workers, it includes three summaries. To reduce the variance between the summaries regarding their coverage of the crisis event described in the tweets that are part of the timeline, we pick the two summaries written by the
two workers who agree the most based on the timeline extraction labels they assign to the tweets in each timeline. This preprocessing step is applied during training and evaluation (for both oracle and non-oracle experiments) and it's done in the [preprocess_timelines.py](timeline-summarization/preprocess_timelines.py) script. It's woroth noting that for the evaluation, this script will generate `dev.sum.json` and `test.sum.json` for the oracle experiments, and `dev.sum.bert.relevant.json` and `test.sum.bert.relevant.json` for the non-oracle experiments. It's important to note that the output of the best timneline-extraction model is needed in order to generate `dev.sum.bert.relevant.json` and `test.sum.bert.relevant.json for the dev and test sets, respectively.

To run the fine-tuning experiments, you would need to run:

```bash

python run_summarization.py \
    --model_name_or_path model_dir \
    --do_predict \
    --validation_file data/dev.sum.bert.relevant.json \ # To run Oracle, simply change this to data/dev.sum.json
    --test_file  data/dev.sum.bert.relevant.json \ # To run Oracle, simply change this to data/dev.sum.json
    --text_column timeline \
    --summary_column summary \
    --per_device_eval_batch_size 16 \
    --output_dir model_dir\
    --num_beams 4 \
    --max_target_length 512 \
    --predict_with_generate \
    --output_file_name dev.sum.bert.relevant.txt \
    --overwrite_cache
```

More details on inference could be found in [generate.sh](timeline-summarization/generate.sh)

#### Evaluation Results on the Test Set:

We used [SacreROUGE](https://github.com/danieldeutsch/sacrerouge) to get multi-reference ROUGE scores. To run the evaluation on the test set, you need to run `python timeline-summarization/evaluate_rouge.py --gold data/test.gold.sum.json --preds /path/to/generated_summaries.txt`.


|              | R1   |  R2  |  RL  |
| -----------  | ----- | ---- | ---- |
| **BART**         |   47.05    |  25.40    |  35.90    |

## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more information.



## Citation:
If you find the code or data in this repo helpful, please cite [our paper](https://arxiv.org/pdf/2210.14190.pdf):
```bibtex
@inproceedings{rajaby-faghihi-etal-2022-crisiltlsum,
    title = "CrisisLTLSum: A Benchmark for Local Crisis Event Timeline Extraction
and Summarization",
    author = "Rajaby Faghihi, Hossein  and
      Alhafni, Bashar  and
      Zhang, Ke  and
      Ran, Shihao  and
      Tetreault, Joel  and
      Jaimes, Alejandro",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Online and Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    abstract = "Social media has increasingly played a key role in emergency response: first responders can use public posts to better react to ongoing crisis events and deploy the necessary resources where they are most needed. Timeline extraction and abstractive summarization are critical technical tasks to leverage large numbers of social media posts about events. Unfortunately, there are few datasets for benchmarking technical approaches for those tasks. This paper presents CrisisLTLSum, the largest dataset of local crisis event timelines available to date. CrisisLTLSum contains 1,000 crisis event timelines across four domains: wildfires, local fires, traffic, and storms. We built CrisisLTLSum using a semi-automated clusterthen-refine approach to collect data from the public Twitter stream. Our initial experiments indicate a significant gap between the performance of strong baselines compared to the human performance on both tasks. Our dataset, code, and models are publicly available",
}

```

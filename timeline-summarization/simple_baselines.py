import numpy as np
import json
import re
import random


def read_data(path):
    with open(path, mode='r') as f:
        raw_timelines = [json.loads(x) for x in f.readlines()]
    return raw_timelines


def first_tweet_baseline(data):
    summaries = []
    for timeline in data:
        summaries.append(clean_text(timeline['tweets'][0]))
    return summaries


def last_tweet_baseline(data):
    summaries = []
    for timeline in data:
        summaries.append(clean_text(timeline['tweets'][-1]))
    return summaries

def random_tweet_baseline(data):
    summaries = []
    for timeline in data:
        summaries.append(clean_text(random.choice(timeline['tweets'])))
    return summaries

def clean_text(text):
    """Removes URLs from the tweet"""
    c_text = re.sub(r'http\S+', '', text).strip()
    c_text = re.sub('\s+', ' ', c_text).strip()
    return c_text

def write_data(summaries, path):
    with open(path, mode='w') as f:
        for summary in summaries:
            f.write(summary)
            f.write('\n')


if __name__ == '__main__':
    data = read_data('/scratch/ba63/CrisisLTLSum/data/dev.json')
    first_tweet_summaries = first_tweet_baseline(data)
    last_tweet_summaries = last_tweet_baseline(data)
    random_tweet_summaries = random_tweet_baseline(data)

    assert len(first_tweet_summaries) == len(last_tweet_summaries) == len(random_tweet_summaries) == len(data)
    write_data(first_tweet_summaries, '/scratch/ba63/CrisisLTLSum/timeline-summarization/baselines/first_tweet_baseline.dev.txt')
    write_data(last_tweet_summaries, '/scratch/ba63/CrisisLTLSum/timeline-summarization/baselines/last_tweet_baseline.dev.txt')
    write_data(random_tweet_summaries, '/scratch/ba63/CrisisLTLSum/timeline-summarization/baselines/random_tweet_baseline.dev.txt')

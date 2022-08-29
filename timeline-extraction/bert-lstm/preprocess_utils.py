from torch.utils.data import Dataset
import re
import json
import torch

class TimelinePreprocess:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label2id_map = {'0': 0, '1': 1}

    def preprocess_lstm(self, timeline):
        """
        Takes a timeline and featurize it as follows
            - seed [SEP] tweet
        """
        batch_id = timeline['batch_id']
        timeline_id = timeline['timeline_id']
        seed = timeline['seed']
        timeline_tweets = timeline['tweets']
        timeline_labels = timeline['labels']

        
        processed_timeline = [self.clean_tweet(seed)]
        label_ids = [-100] # ignore label for seed

        for tweet, label in zip(timeline_tweets, timeline_labels):
            clean_tweet = self.clean_tweet(tweet)

            processed_timeline.append(clean_tweet)
            label_ids.append(self.label2id_map[label])

        assert len(processed_timeline) == len(label_ids)

        # we will feed the seeds and the tweets as pairs and let the tokenizer
        # handle the different segment ids
        features = self.tokenizer(processed_timeline, padding=True, max_length=self.max_seq_length,
                                  truncation=True, return_tensors='pt')

        features['label_ids'] = torch.LongTensor(label_ids)

        return features

    def clean_tweet(self, tweet):
        """Removes URLs from the tweet"""
        c_tweet = re.sub(r'http\S+', '', tweet).strip()
        c_tweet = re.sub('\s+', ' ', c_tweet).strip()
        return c_tweet

class TimelineDataset(Dataset):
    def __init__(self, examples, preprocessor):
        self.examples = examples
        self.preprocessor = preprocessor

    @classmethod
    def create_dataset(cls, path, preprocessor):
        with open(path, mode='r') as f:
            raw_timelines = [json.loads(x) for x in f.readlines()]

        featurized_examples = []
        for timeline in raw_timelines:

            featurized_timeline = preprocessor.preprocess_lstm(timeline)

            featurized_examples.append(featurized_timeline)

        return cls(featurized_examples, preprocessor)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

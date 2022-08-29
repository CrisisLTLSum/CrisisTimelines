from torch.utils.data import Dataset
import re
import json


class TimelinePreprocess:
    def __init__(self, tokenizer, max_seq_length, relevant_tweets_only=False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.relevant_tweets_only = relevant_tweets_only
        self.label2id_map = {'0': 0, '1': 1}

    def preprocess_bert(self, timeline):
        """
        Takes a timeline and featurize it as follows
            - seed [SEP] context; where the context will always contain the
                current tweet and the relevant tweets before it
        """
        batch_id = timeline['batch_id']
        timeline_id = timeline['timeline_id']
        seed = timeline['seed']
        timeline_tweets = timeline['tweets']
        timeline_labels = timeline['labels']

        context = []
        label_ids = []
        processed_timeline = []

        for tweet, label in zip(timeline_tweets, timeline_labels):

            clean_tweet = self.clean_tweet(tweet)
            clean_seed = self.clean_tweet(seed)

            processed_timeline.append((clean_seed, " ".join(context +  [clean_tweet])))
            # processed_timeline.append(clean_seed + "[SEP]" + "[SEP]".join(context) + "[SEP]" + clean_tweet)

            label_ids.append(self.label2id_map[label])
            if self.relevant_tweets_only:
                if label == '1':
                    context.append(clean_tweet)
            else:
                context.append(clean_tweet)

        # import pdb; pdb.set_trace()
        assert len(processed_timeline) == len(timeline_tweets) == len(label_ids)

        all_seeds = [x[0] for x in processed_timeline]
        all_tweets = [x[1] for x in processed_timeline]
        assert len(set(all_seeds)) == 1

        # we will feed the seeds and the tweets as pairs and let the tokenizer
        # handle the different segment ids
        features = self.tokenizer(all_seeds, all_tweets, max_length=self.max_seq_length,
                                  truncation='only_second')

        # features = self.tokenizer(processed_timeline, max_length=self.max_seq_length, truncation=True)

        features['label_ids'] = label_ids

        return features

    def preprocess_bert_simple(self, timeline):
        """
        Takes a timeline and featurize it as follows
            - seed [SEP] tweet
        """
        batch_id = timeline['batch_id']
        timeline_id = timeline['timeline_id']
        seed = timeline['seed']
        timeline_tweets = timeline['tweets']
        timeline_labels = timeline['labels']

        label_ids = []
        processed_timeline = []

        for tweet, label in zip(timeline_tweets, timeline_labels):
            clean_tweet = self.clean_tweet(tweet)
            clean_seed = self.clean_tweet(seed)

            processed_timeline.append((clean_seed, clean_tweet))
            # processed_timeline.append(clean_seed + "[SEP]" + "[SEP]".join(context) + "[SEP]" + clean_tweet)

            label_ids.append(self.label2id_map[label])

        assert len(processed_timeline) == len(timeline_tweets) == len(label_ids)

        all_seeds = [x[0] for x in processed_timeline]
        all_tweets = [x[1] for x in processed_timeline]
        assert len(set(all_seeds)) == 1

        # we will feed the seeds and the tweets as pairs and let the tokenizer
        # handle the different segment ids
        features = self.tokenizer(all_seeds, all_tweets, max_length=self.max_seq_length,
                                  truncation='only_second')

        # features = self.tokenizer(processed_timeline, max_length=self.max_seq_length, truncation=True)

        features['label_ids'] = label_ids

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
            featurized_timeline = preprocessor.preprocess_bert(timeline)
            # featurized_timeline = preprocessor.preprocess_bert_simple(timeline)
            for i in range(len(featurized_timeline['input_ids'])):
                featurized_examples.append({'input_ids': featurized_timeline['input_ids'][i],
                                            'attention_mask': featurized_timeline['attention_mask'][i],
                                            'token_type_ids': featurized_timeline['token_type_ids'][i],
                                            'label': featurized_timeline['label_ids'][i]})

        return cls(featurized_examples, preprocessor)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)
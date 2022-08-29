from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn.functional as F
import re
import json
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score


class RelevanceClassifier:
    def __init__(self, model_path, max_seq_length):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.label2id = {'0': 0, '1': 1}
        self.max_seq_length = max_seq_length
        self.model.eval()

    def predict_timeline(self, seed, tweets):

        predictions = []
        context = []

        for tweet in tweets:
            # import pdb; pdb.set_trace()
            c_tweet = clean_tweet(tweet)
            clean_seed = clean_tweet(seed)

            input_tweet = " ".join(context + [c_tweet])
            # input_tweet = clean_seed + "[SEP]" + "[SEP]".join(context) + "[SEP]" + c_tweet

            with torch.no_grad():
                inputs = self.convert_example_to_features(clean_seed, input_tweet)
                # inputs = self.convert_example_to_features(input_tweet)

                logits = self.model(**inputs)[0]
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()

            predictions.append(pred)
            # add the current tweet to the context only if it's relevant
            if pred == 1:
                context.append(c_tweet)

        assert len(predictions) == len(tweets)

        return predictions

    def predict(self, data):
        data_with_preds = []

        for timeline in data:

            seed = timeline['seed']
            tweets = timeline['tweets']
            timeline_preds = self.predict_timeline(seed, tweets)

            timeline['predictions'] = timeline_preds

            data_with_preds.append(timeline)

        return data_with_preds

    def convert_example_to_features(self, seed, tweet):
        features = self.tokenizer(seed, tweet,
                                  padding=False, max_length=self.max_seq_length,
                                  truncation='only_second',
                                  return_tensors="pt")
        return features

def clean_tweet(tweet):
    """Removes URLs from the tweet"""
    c_tweet = re.sub(r'http\S+', '', tweet).strip()
    c_tweet = re.sub('\s+', ' ', c_tweet).strip()
    return c_tweet

def read_data(path):
    with open(path, mode='r') as f:
        data = [json.loads(x) for x in f.readlines()]
    return data

def compute_results(preds, path):
    all_gold = []
    all_preds = []

    for timeline in preds:
        preds = np.asarray(timeline['predictions'])
        gold = np.asarray([int(l) for l in timeline['labels']])

        all_gold.append(gold)
        all_preds.append(preds)

    results = compute_metrics(all_gold, all_preds)

    with open(path, mode='w') as f:
        json.dump(results, f)

    return results

def write_predictions(data, path):
    with open(path, mode='w') as f:
        for timeline in data:
            json.dump(timeline, f, ensure_ascii=False)
            f.write('\n')

def compute_metrics(gold, preds):
    # both gold and preds are list of lists where each list represents a timeline
    assert len(gold) == len(preds)
    all_gold = np.asarray([l for labels in gold for l in labels])
    all_preds = np.asarray([l for labels in preds for l in labels])

    # tweet-level metrics
    overall_acc = (all_gold == all_preds).mean()
    overall_f1 = f1_score(y_true=all_gold, y_pred=all_preds, average='macro')
    overall_precision = precision_score(y_true=all_gold, y_pred=all_preds, average='macro')
    overall_recall = recall_score(y_true=all_gold, y_pred=all_preds, average='macro')

    #timeline-level metrics
    avg_timeline_acc, avg_timeline_f1, avg_timeline_recall, avg_timeline_precision = [], [], [], []

    for y_gold, y_pred in zip(gold, preds):
        y_gold = np.asarray(y_gold)
        y_pred = np.asarray(y_pred)

        timeline_acc = (y_gold == y_pred).mean()
        timeline_f1 = f1_score(y_true=y_gold, y_pred=y_pred, average='macro')
        timeline_precision = precision_score(y_true=y_gold, y_pred=y_pred, average='macro')
        timeline_recall =  recall_score(y_true=y_gold, y_pred=y_pred, average='macro')

        avg_timeline_acc.append(timeline_acc)
        avg_timeline_f1.append(timeline_f1)
        avg_timeline_recall.append(timeline_recall)
        avg_timeline_precision.append(timeline_precision)

    avg_timeline_acc = np.asarray(avg_timeline_acc).mean()
    avg_timeline_f1 = np.asarray(avg_timeline_f1).mean()
    avg_timeline_recall = np.asarray(avg_timeline_recall).mean()
    avg_timeline_precision = np.asarray(avg_timeline_precision).mean()

    return {'Avg Timeline Accuracy': avg_timeline_acc,
            'Avg Timeline F1': avg_timeline_f1,
            'Avg Timeline Precision': avg_timeline_recall,
            'Avg Timeline Recall': avg_timeline_precision,
            'Overall Accuracy': overall_acc,
            'Overall F1': overall_f1,
            'Overall Precision': overall_precision,
            'Overall Recall': overall_recall
            }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--max_seq_length', type=int, required=True)
    args = parser.parse_args()

    data = read_data(args.eval_data_file)
    classifier = RelevanceClassifier(args.model_path, max_seq_length=args.max_seq_length)
    data_with_preds = classifier.predict(data)
    results = compute_results(data_with_preds, path=args.model_path+'/prediction_results_test.txt')
    write_predictions(data_with_preds, path=args.model_path+'/predictions_test.json')
    print(results)

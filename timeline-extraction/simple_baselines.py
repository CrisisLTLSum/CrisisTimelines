import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score


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


def read_data(path):
    with open(path, mode='r') as f:
        raw_timelines = [json.loads(x) for x in f.readlines()]
    return raw_timelines

def simple_baseline(data, label=0):
    gold, preds = [], []
    for i, x in enumerate(data):
        labels = [int(l) for l in x['labels']]
        baseline_preds = np.asarray([label] * len(x['labels']))

        gold.append(labels)
        preds.append(baseline_preds)

    return compute_metrics(gold, preds)

def get_majority_baseline(data):
    all_labels = []
    for timeline in data:
        labels = np.asarray([int(x) for x in timeline['labels']])
        all_labels.extend(labels)

    majority_label = np.argmax(np.bincount(all_labels))
    return majority_label


if __name__ == '__main__':
    train_dataset = read_data('/home/balhafni/timeline-summarization/new_data/train.json')
    dev_dataset = read_data('/home/balhafni/timeline-summarization/new_data/dev.json')

    all_0 = simple_baseline(dev_dataset, 0)
    all_1 = simple_baseline(dev_dataset, 1)

    print('Results if everything was labeled as 0 baseline: ')
    print(all_0)
    print()
    print('Results if everything was labeled as 1 baseline:')
    print(all_1)
    print()

    majority_class = get_majority_baseline(train_dataset)
    majority = simple_baseline(dev_dataset, majority_class)
    print(f'Results of majority baseline ({majority_class}): ')
    print(majority)


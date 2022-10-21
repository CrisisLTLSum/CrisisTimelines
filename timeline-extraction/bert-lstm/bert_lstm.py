import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from preprocess_utils import TimelinePreprocess, TimelineDataset
import random
import numpy as np
import json
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score

class RelevanceClassifier(nn.Module):
    def __init__(self, bert_model_path, hidden_size, num_layers, num_labels, dropout, device='cuda'):
        super(RelevanceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_labels)
        self.device = device

        # freeze bert
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, timelines, lengths):

        bert_embeddings, labels = self.get_bert_embeddings(timelines)

        packed_embedded_seqs = pack_padded_sequence(bert_embeddings, lengths,
                                                    batch_first=True)

        output, h_n = self.rnn(packed_embedded_seqs)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        # output shape: (batch_size, num_tweets, num_dirs * hidden_size)

        logits = self.output_layer(output)
        # logits shape: (batch_size, num_tweets, 2)
        return logits, labels

    def get_bert_embeddings(self, timelines):

        all_embeddings = []
        all_labels = []
        # get the [CLS] representation of each timeline
        for timeline in timelines:
            timeline = {k: v.to(self.device) for k, v in timeline.items()}
            inputs = {'input_ids': timeline['input_ids'],
                      'attention_mask': timeline['attention_mask'],
                      'token_type_ids': timeline['token_type_ids']}

            labels = timeline['label_ids'].to(self.device)

            bert_embeddings = self.bert(**inputs).pooler_output
            # bert_embeddings.shape: (num_tweets, 768)
            all_embeddings.append(bert_embeddings)
            all_labels.append(labels)

        all_embeddings = pad_sequence(all_embeddings, batch_first=True, padding_value=0)
        # all_embeddings.shape: (batch_size, num_tweets, 768)
        all_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
        # all_labels.shape: (batch_size, num_tweets)

        return all_embeddings, all_labels

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        timelines = batch['timelines']
        lengths = batch['lengths']

        logits, labels = model(timelines, lengths)

        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            timelines = batch['timelines']
            lengths = batch['lengths']

            logits, labels = model(timelines, lengths)

            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def predict(model, dataloader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            timelines = batch['timelines']
            lengths = batch['lengths']

            logits, labels = model(timelines, lengths)
            predictions, clean_labels = align_predictions(logits.to('cpu'), labels.to('cpu'))
            all_predictions.extend(predictions)
            all_labels.extend(clean_labels)

    return all_predictions, all_labels

def align_predictions(predictions,
                        label_ids):

    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    label_map = {0: 0, 1: 1}

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j].item()])
                preds_list[i].append(label_map[preds[i][j].item()])

        assert len(out_label_list[i]) == len(preds_list[i])

    return preds_list, out_label_list

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def read_data(path):
    with open(path, mode='r') as f:
        data = [json.loads(x) for x in f.readlines()]
    return data

class Collator:
    def __init__(self, tokenizer, max_length, pad_to_multiple_of):
        self.tokenizer = tokenizer
        self.padding = True
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = "pt"

    def __call__(self, features):

        # sorting the batch in descending order based on length of the timelines
        sorted_features = sorted(features, key=lambda x: len(x['input_ids']),
                                 reverse=True)

        lengths = torch.LongTensor([len(x['input_ids']) for x in sorted_features])

        return {'timelines': sorted_features, 'lengths': lengths}

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
    parser.add_argument('--bert_model', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--hidd_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_predict', action="store_true")
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()

    set_seed(args.seed, True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    data_processor = TimelinePreprocess(tokenizer, max_seq_length=tokenizer.model_max_length)

    train_dataset = TimelineDataset.create_dataset(path=args.train_file,
                                                   preprocessor=data_processor)

    dev_dataset = TimelineDataset.create_dataset(path=args.dev_file,
                                                 preprocessor=data_processor)

    collator = Collator(tokenizer, max_length=tokenizer.model_max_length, pad_to_multiple_of=1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)


    model = RelevanceClassifier(bert_model_path=args.bert_model, hidden_size=args.hidd_size,
                                num_layers=args.num_layers, num_labels=2, dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.84, 1.2], device=device), ignore_index=-100)
    model = model.to(device)

    train_losses = []
    eval_losses = []

    best_dev_acc = 0
    patience = 5
    early_stopping = 0

    if args.do_train:
        print(f'Training...')
        for epoch in range(args.num_epochs):
            set_seed(args.seed, True)
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            dev_loss = evaluate(model, dev_dataloader, criterion, device)

            dev_preds, ground_truth = predict(model, dev_dataloader)

            dev_results = compute_metrics(dev_preds, ground_truth)

            # save the best checkpoint based on avg timeline accuracy
            if dev_results['Avg Timeline Accuracy'] > best_dev_acc:
                early_stopping = 0
                best_dev_acc = dev_results['Avg Timeline Accuracy']
                torch.save(model.state_dict(), f'/scratch/ba63/CrisisLTLSum/timeline-extraction/bert-lstm/checkpoints/checkpoint-best')
            else:
                early_stopping += 1

            train_losses.append(train_loss)
            eval_losses.append(dev_loss)

            print(f'Epoch: {(epoch + 1)}')
            torch.save(model.state_dict(), f'/scratch/ba63/CrisisLTLSum/timeline-extraction/bert-lstm/checkpoints/checkpoint-{epoch + 1}')
            print(f'\tTrain Loss: {train_loss:.4f}   |   Dev Loss: {dev_loss:.4f}   |   Best Acc: {best_dev_acc:.4f}')

            if early_stopping > patience:
                print(f"Dev acc hasn't increased in {patience} epochs .. stopping training")
                break

    if args.do_predict:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        predictions, ground_truth = predict(model, dev_dataloader)

        results = compute_metrics(predictions, ground_truth)
        print(results)
        with open(args.checkpoint + '.prediction_results.txt', mode='w') as f:
            json.dump(results, f)

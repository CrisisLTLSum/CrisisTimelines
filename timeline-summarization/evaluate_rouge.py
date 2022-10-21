import json
from sacrerouge.metrics import Rouge
import argparse


def read_gold(path):
    with open(path, mode='r') as f:
        lines = [json.loads(x) for x in f.readlines()]
    references = []
    for line in lines:
        references.append([line['reference-1'], line['reference-2']])
    return references

def read_preds(path):
    with open(path, mode='r') as f:
        preds = [line.strip() for line in f.readlines()]
        return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold')
    parser.add_argument('--preds')

    args = parser.parse_args()

    # gold = read_gold(args.gold)[:132]
    # preds = read_preds(args.preds)[:132]
    gold = read_gold(args.gold)
    preds = read_preds(args.preds)
    assert len(preds) == len(gold)

    rouge = Rouge(compute_rouge_l=True)
    rouge_score = rouge.evaluate(preds, gold)
    serialized_score = json.dumps(rouge_score[0], indent=2)
    print(serialized_score)

    with open(args.preds+'.results', mode='w') as f:
        f.write(serialized_score)

    # with open(args.preds+'.results_132', mode='w') as f:
    #     f.write(serialized_score)

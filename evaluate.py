import os
import sys
import json
from datetime import date
from collections import defaultdict, Counter
from tqdm import tqdm
def whether_equal(answer, pred):
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def load(f):
    data = []
    for line in f:
        data.append(json.loads(line.strip()))
    return data
def main():
    gt_folder, pred_fn = sys.argv[1], sys.argv[2]

    gt_fn = os.path.join(gt_folder, 'test_answer.json')
    gt = json.load(open(gt_fn))
    pred = [x.strip() for x in open(pred_fn).readlines()] # one prediction per line
    train_set = json.load(open(os.path.join(gt_folder, 'train.json')))
    train_answer_set = set(x['answer'] for x in train_set)

    labels = ['overall', 'multihop', 'qualifier', 'comparison', 'logical', 'count', 'verify', 'zero-shot']
    total = {k:0 for k in labels}
    correct = {k:0 for k in labels}
    for i in tqdm(range(len(pred))):
        cur_labels = ['overall']
        functions = [f['function'] for f in gt[i]['program']]

        for f in functions:
            if f in {'Relate'} or f.startswith('Filter'):
                cur_labels.append('multihop')
                break
        for f in functions:
            if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
                cur_labels.append('qualifier')
                break
        for f in functions:
            if f in {'SelectBetween','SelectAmong'}:
                cur_labels.append('comparison')
                break
        for f in functions:
            if f in {'And', 'Or'}:
                cur_labels.append('logical')
                break
        for f in functions:
            if f in {'Count'}:
                cur_labels.append('count')
                break
        for f in functions:
            if f in {'VerifyStr','VerifyNum','VerifyYear','VerifyDate'}:
                cur_labels.append('verify')
                break

        answer = gt[i]['answer']
        if answer not in train_answer_set:
            cur_labels.append('zero-shot')

        if whether_equal(answer, pred[i]):
            for k in cur_labels:
                correct[k] += 1
        else:
            pass
        for k in cur_labels:
            total[k] += 1

    for k in labels:
        print('{}: {:.2f}% ({}/{})'.format(k, correct[k]/total[k]*100, correct[k], total[k]))
    if len(pred) < len(gt):
        print('WARNING: there are only {} predictions (need {})'.format(len(pred), len(gt)))


if __name__ == '__main__':
    main()

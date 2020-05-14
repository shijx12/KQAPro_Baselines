import sys
import json
from collections import defaultdict, Counter


def main():
    gt_fn, pred_fn = sys.argv[1], sys.argv[2]

    gt = json.load(open(gt_fn))
    pred = [x.strip() for x in open(pred_fn).readlines()] # one prediction per line

    labels = ['overall', 'multihop', 'highlevel', 'comparison', 'logical', 'count', 'verify']
    total = {k:0 for k in labels}
    correct = {k:0 for k in labels}
    for i in range(len(pred)):
        cur_labels = ['overall']
        functions = [f['function'] for f in gt[i]['program']]

        for f in functions:
            if f in {'Relate'} or f.startswith('Filter'):
                cur_labels.append('multihop')
                break
        for f in functions:
            if f in {'QFilterStr', 'QFilterNum', 'QFilterYear', 'QFilterDate', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier'}:
                cur_labels.append('highlevel')
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
        if answer == pred[i]:
            for k in cur_labels:
                correct[k] += 1
        else:
            print(i)
        for k in cur_labels:
            total[k] += 1

    for k in labels:
        print('{}: {:.2f}% ({}/{})'.format(k, correct[k]/total[k]*100, correct[k], total[k]))
    if len(pred) < len(gt):
        print('WARNING: there are only {} predictions (need {})'.format(len(pred), len(gt)))


if __name__ == '__main__':
    main()

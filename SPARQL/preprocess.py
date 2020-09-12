"""
We need the last function to help extract the final answer of SPARQL, used in check_sparql
"""

import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re

from utils.misc import init_vocab

def tokenize_sparql(s): 
    # separate punctuations
    s = s.replace('"', ' " ').replace('^^', ' ^^ ')
    # NOTE: after decoding, these extra space must be removed
    # this may cause some mistakes, but the ratio is very small, about one of thousands
    return s.split()

def postprocess_sparql_tokens(s):
    # organize the predicted sparql tokens into a valid query
    s = s.replace(' ^^ ', '^^')
    skip_idxs = set()
    for i in range(len(s)):
        if s[i] == '"':
            if i > 2 and s[i-1]==' ' and s[i-2] not in {'>'}:
                skip_idxs.add(i-1)
            if i < len(s)-2 and s[i+1]==' ' and s[i+2] not in {'<'}:
                skip_idxs.add(i+1)
    s = ''.join([s[i] for i in range(len(s)) if i not in skip_idxs])
    return s

def encode_dataset(dataset, vocab, test=False):
    questions = []
    sparqls = []
    choices = []
    answers = []
    for question in tqdm(dataset):
        q = [vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) 
            for w in word_tokenize(question['question'].lower())]
        questions.append(q)

        _ = [vocab['answer_token_to_idx'][w] for w in question['choices']]
        choices.append(_)

        if test:
            continue

        _ = [vocab['sparql_token_to_idx'].get(w, vocab['sparql_token_to_idx']['<UNK>']) 
            for w in tokenize_sparql(question['sparql'])]
        # wrap with <START> <END>
        _ = [vocab['sparql_token_to_idx']['<START>']] + _ + [vocab['sparql_token_to_idx']['<END>']]
        sparqls.append(_)

        if 'answer' in question:
            answers.append(vocab['answer_token_to_idx'].get(question['answer']))

    # question padding
    max_len = max(len(q) for q in questions)
    for q in questions:
        while len(q) < max_len:
            q.append(vocab['word_token_to_idx']['<PAD>'])
    if not test:
        # sparql padding
        max_len = max(len(s) for s in sparqls)
        for s in sparqls:
            while len(s) < max_len:
                s.append(vocab['sparql_token_to_idx']['<PAD>'])

    questions = np.asarray(questions, dtype=np.int32)
    sparqls = np.asarray(sparqls, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    return questions, sparqls, choices, answers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_cnt', type=int, default=1)
    args = parser.parse_args()


    print('Build kb vocabulary')
    vocab = {
        'word_token_to_idx': init_vocab(),
        'sparql_token_to_idx': init_vocab(),
        'answer_token_to_idx': {}
    }
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))
    val_set = json.load(open(os.path.join(args.input_dir, 'val.json')))
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    print('Build question vocabulary')
    word_counter = Counter()
    for question in train_set:
        tokens = word_tokenize(question['question'].lower())
        word_counter.update(tokens)
        # add candidate answers
        for a in question['choices']:
            if a not in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])
        # add sparql
        for a in tokenize_sparql(question['sparql']):
            if a not in vocab['sparql_token_to_idx']:
                vocab['sparql_token_to_idx'][a] = len(vocab['sparql_token_to_idx'])

    # filter low-frequency words
    for w, c in word_counter.items():
        if w and c >= args.min_cnt and w not in vocab['word_token_to_idx']:
            vocab['word_token_to_idx'][w] = len(vocab['word_token_to_idx'])
    # add candidate answers of val and test set
    for question in chain(val_set, test_set):
        for a in question['choices']:
            if a not in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])


    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, vocab, name=='test')
        assert len(outputs) == 4
        print('shape of questions, sparqls, choices, answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)





if __name__ == '__main__':
    main()

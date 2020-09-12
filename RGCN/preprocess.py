import os
import json
import pickle
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import argparse

from utils.load_kb import load_as_graph
from utils.misc import init_vocab


def encode_dataset(dataset, vocab):
    questions = []
    choices = []
    answers = []
    for question in tqdm(dataset):
        q = [vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) 
            for w in word_tokenize(question['question'].lower())]
        questions.append(q)

        _ = [vocab['answer_token_to_idx'][w] for w in question['choices']]
        choices.append(_)

        if 'answer' in question:
            answers.append(vocab['answer_token_to_idx'].get(question['answer']))

    # question padding
    max_len = max(len(q) for q in questions)
    for q in questions:
        while len(q) < max_len:
            q.append(vocab['word_token_to_idx']['<PAD>'])

    questions = np.asarray(questions, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    return questions, choices, answers




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_cnt', type=int, default=1)
    parser.add_argument('--max_desc', type=int, default=200)
    args = parser.parse_args()


    print('Load and encode kb...')
    kb_vocab, node_descs, triples, nodeid2idx, pred2idx = \
        load_as_graph(os.path.join(args.input_dir, 'kb.json'), args.max_desc)
    node_descs = np.asarray(node_descs)
    triples = np.asarray(triples)
    print("shape of node_descs and triples:", node_descs.shape, triples.shape)
    print(node_descs[-10:])
    print(triples[:10])

    with open(os.path.join(args.output_dir, 'kb.pt'), 'wb') as f:
        pickle.dump(node_descs, f)
        pickle.dump(triples, f)


    vocab = {
        'kb_token_to_idx': kb_vocab,
        'predicate_token_to_idx': pred2idx,
        'word_token_to_idx': init_vocab(),
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
        outputs = encode_dataset(dataset, vocab)
        print('shape of questions, choices, answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)




if __name__ == '__main__':
    main()


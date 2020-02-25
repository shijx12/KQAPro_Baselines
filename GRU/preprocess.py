import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm

from utils import get_kb_vocab, init_vocab


def tokenize_sparql(s): 
    # TODO
    return s.split()

def encode_dataset(dataset, vocab, test=False):
    questions = []
    functions = []
    func_depends = []
    func_inputs = []
    sparqls = []
    choices = []
    answers = []
    for question in tqdm(dataset):
        q = [vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) 
            for w in word_tokenize(question['text'].lower())]
        questions.append(q)

        _ = [vocab['answer_token_to_idx'][w] for w in question['choices']]
        choices.append(_)

        if test:
            continue

        func, dep, inp = [], [], []
        for f in question['program']:
            func.append(vocab['function_token_to_idx'][f['function']])
            dep.append(f['dependencies'])
            inp.append([vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) 
                    for w in f['inputs']])
        functions.append(func)
        func_depends.append(dep)
        func_inputs.append(inp)

        _ = [vocab['sparql_token_to_idx'].get(w, vocab['sparql_token_to_idx']['<UNK>']) 
            for w in tokenize_sparql(question['sparql'])]
        sparqls.append(_)

        if 'answer' in question:
            answers.append(vocab['answer_token_to_idx'].get(question['answer']))

    # question padding
    max_len = max(len(q) for q in questions)
    for q in questions:
        while len(q) < max_len:
            q.append(vocab['word_token_to_idx']['<PAD>'])
    if not test:
        # function padding
        max_len = max(len(f) for f in functions)
        max_dep = 2
        max_inp = 0
        for inp in func_inputs:
            max_inp = max(max_inp, max(len(i) for i in inp))
        for i in range(len(functions)):
            for j in range(len(functions[i])):
                func_depends[i][j] += [-1] * (max_dep - len(func_depends[i][j])) # use -1 to pad dependency
                func_inputs[i][j] += [vocab['word_token_to_idx']['<PAD>']] * (max_inp - len(func_inputs[i][j]))
            while len(functions[i]) < max_len:
                functions[i].append(vocab['function_token_to_idx']['<PAD>'])
                func_depends[i].append([-1, -1])
                func_inputs[i].append([vocab['word_token_to_idx']['<PAD>']] * max_inp)
        # sparql padding
        max_len = max(len(s) for s in sparqls)
        for s in sparqls:
            while len(s) < max_len:
                s.append(vocab['sparql_token_to_idx']['<PAD>'])

    questions = np.asarray(questions, dtype=np.int32)
    functions = np.asarray(functions, dtype=np.int32)
    func_depends = np.asarray(func_depends, dtype=np.int32)
    func_inputs = np.asarray(func_inputs, dtype=np.int32)
    sparqls = np.asarray(sparqls, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    return questions, functions, func_depends, func_inputs, sparqls, choices, answers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_cnt', type=int, default=1)
    args = parser.parse_args()


    print('Build kb vocabulary')
    vocab_kb = get_kb_vocab(os.path.join(args.input_dir, 'kb.json'), args.min_cnt)
    vocab = {
        'kb_token_to_idx': vocab_kb,
        'word_token_to_idx': init_vocab(), # include question text and function inputs
        'function_token_to_idx': init_vocab(),
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
        tokens = word_tokenize(question['text'].lower())
        word_counter.update(tokens)
        # add candidate answers
        for a in question['choices']:
            if a not in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])
        # add functions
        for f in question['program']:
            a = f['function']
            if a not in vocab['function_token_to_idx']:
                vocab['function_token_to_idx'][a] = len(vocab['function_token_to_idx'])
            # consider function inputs as words
            word_counter.update(f['inputs'])
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
        print('shape of questions, functions, func_depends, func_inputs, sparqls, choices, answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)





if __name__ == '__main__':
    main()

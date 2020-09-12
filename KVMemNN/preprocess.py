import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm

from utils.load_kb import load_as_key_value
from utils.misc import init_vocab


def create_inverted(keys):
    inverted_index = defaultdict(set)
    counter = Counter()
    for i in range(len(keys)):
        for w in keys[i]:
            inverted_index[w].add(i)
            counter[w] += 1
    return inverted_index


def find_candidate_keys(inverted_index, stopwords, question, num_cand_keys):
    """
    find keys that are relevant to question, and then return the top num_cand_keys
    if not enough, pad 0
    """
    words = word_tokenize(question['question'].lower())
    counter = Counter()
    for w in words:
        if w in stopwords: # skip stopwords
            continue
        counter.update(inverted_index.get(w, []))
    indexes = [x[0] for x in counter.most_common(num_cand_keys)]
    if len(indexes) < num_cand_keys:
        indexes += [0] * (num_cand_keys - len(indexes))
    return indexes



def encode_kb(keys, values, vocab):
    encoded_keys = []
    encoded_values = []
    for i in tqdm(range(len(keys))):
        encoded_keys.append([vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) for w in keys[i]])
        encoded_values.append([vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) for w in values[i]])
    keys = encoded_keys
    values = encoded_values
    max_len = max(len(k) for k in keys)
    for k in keys:
        while len(k) < max_len:
            k.append(vocab['word_token_to_idx']['<PAD>'])
    max_len = max(len(k) for k in values)
    for k in values:
        while len(k) < max_len:
            k.append(vocab['word_token_to_idx']['<PAD>'])
    keys = np.asarray(keys, dtype=np.int32)
    values = np.asarray(values, dtype=np.int32)
    return keys, values


def encode_dataset(dataset, vocab, inverted_index, stopwords, num_cand_keys):
    questions = []
    key_indexes = []
    choices = []
    answers = []
    for question in tqdm(dataset):
        q = [vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>']) 
            for w in word_tokenize(question['question'].lower())]
        questions.append(q)

        key_indexes.append(find_candidate_keys(inverted_index, stopwords, question, num_cand_keys))
        

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
    key_indexes = np.asarray(key_indexes, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    return questions, key_indexes, choices, answers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_cnt', type=int, default=1)
    parser.add_argument('--stop_thresh', type=int, default=1000)
    parser.add_argument('--num_cand_keys', type=int, default=1000)
    args = parser.parse_args()


    print('Build kb vocabulary')
    kb_vocab, kb_keys, kb_values = load_as_key_value(os.path.join(args.input_dir, 'kb.json'), args.min_cnt)
    vocab = {
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
    # filter low-frequency words
    stopwords = set()
    for w, c in word_counter.items():
        if w and c >= args.min_cnt and w not in vocab['word_token_to_idx']:
            vocab['word_token_to_idx'][w] = len(vocab['word_token_to_idx'])
        if w and c >= args.stop_thresh:
            stopwords.add(w)
    print('number of stop words (>={}): {}'.format(args.stop_thresh, len(stopwords)))
    # merge kb vocab
    for w in kb_vocab:
        if w not in vocab['word_token_to_idx']:
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

    print('Create inverted index for keys')
    inverted_index = create_inverted(kb_keys)

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, vocab, inverted_index, stopwords, args.num_cand_keys)
        print('shape of questions, key indexes, choices, answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)

    print('Encode kb')
    outputs = encode_kb(kb_keys, kb_values, vocab)
    print('shape of keys, values:')
    with open(os.path.join(args.output_dir, 'kb.pt'), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)




if __name__ == '__main__':
    main()

import os
import json
import pickle
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re
from stanfordcorenlp import StanfordCoreNLP
from torch.utils.data import DataLoader, Dataset
import torch

current_path = os.getcwd()
previous_path = os.path.abspath(os.path.dirname(os.getcwd()))
input_path = previous_path + '/test_dataset'
output_path = previous_path + '/preprocess'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

def load_kg(file_path):
    print('Build kb vocabulary')

    with open(previous_path + '/test_dataset/kb.json', 'r') as f:
        kb = json.load(f)

    # construct vocab
    ent_vocab, rel_vocab = [{} for _ in range(2)]
    triples = []
    for i in kb['entities']:
        for j in kb['entities'][i]['instanceOf']:
            s = kb['entities'][i]['name']
            o = kb['concepts'][j]['name']
            ent_vocab[s] = ent_vocab.get(s, 0) + 1
            ent_vocab[o] = ent_vocab.get(o, 0) + 1
        name = kb['entities'][i]['name']
        for rel_dict in kb['entities'][i]['relations']:
            rel_vocab[rel_dict['predicate']] = rel_vocab.get(rel_dict['predicate'], 0) + 1
            try:
                triples.append([name, rel_dict['predicate'], rel_dict['direction'],
                                kb['entities'][rel_dict['object']]['name']])
            except:
                continue

    entity_vocab, relation_vocab = [{'<PAD>': 0} for _ in range(2)]
    for i, key in enumerate(ent_vocab):
        entity_vocab[key] = len(entity_vocab)
    for i, key in enumerate(rel_vocab):
        relation_vocab[key] = len(rel_vocab)
    with open(file_path + '/entity_vocab.pkl', 'wb') as f:
        pickle.dump(entity_vocab, f)
    with open(file_path + '/relation_vocab.pkl', 'wb') as f:
        pickle.dump(relation_vocab, f)

    # construct triples
    triples_new = []
    for i in range(len(triples)):
        if triples[i][0] == triples[i][-1]:
            continue
        if triples[i][2] == 'forward':
            triples_new.append(triples[i][0] + '\t' + triples[i][1] + '\t' + triples[i][-1])
        elif triples[i][2] == 'backward':
            triples_new.append(triples[i][-1] + '\t' + triples[i][1] + '\t' + triples[i][0])
    triples_new = list(set(triples_new))
    with open(file_path + '/triples.pkl', 'wb') as f:
        pickle.dump(triples_new, f)

    return entity_vocab, relation_vocab, triples_new

def golve_attr(text, pre_train):
    dim = len(pre_train['the'])
    attr = np.zeros((dim,))
    tokens = text.split()
    if len(tokens) > 1:
        for token in tokens:
            attr = attr + pre_train.get(token, pre_train['the'])
        attr = attr/ len(tokens)
    else:
        attr = pre_train.get(tokens[0], pre_train['the'])
    return list(attr)

def build_gcn(entity_vocab, triples):
    with open(output_path + '/glove.840B.300d.py36.pt', 'rb') as f:
        glove = pickle.load(f)
    edge_index, edge_attr = [[] for _ in range(2)]
    for i in tqdm(range(len(triples))):
        t = triples[i].split('\t')
        edge_index.append([entity_vocab[t[0]], entity_vocab[t[2]]])
        edge_attr.append(sum(golve_attr(t[1], glove)))
    entity_attr = [golve_attr(k, glove) for k in entity_vocab]
    gcn_input = {'edge_index': edge_index, 'edge_attr': edge_attr, 'entity_attr': entity_attr}
    with open(output_path + '/gcn_input.pkl', 'wb') as f:
        pickle.dump(gcn_input, f)

def build_vocab():
    print('Build question vocabulary')

    vocab = {
        'word_token_to_idx': {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3},
        'answer_token_to_idx': {}
    }
    train_set = json.load(open(os.path.join(input_path, 'train.json')))
    val_set = json.load(open(os.path.join(input_path, 'val.json')))
    test_set = json.load(open(os.path.join(input_path, 'test.json')))
    stop_thresh = 1000
    min_cnt = 1

    word_counter = Counter()
    for question in tqdm(train_set):
        tokens = word_tokenize(question['text'].lower())
        word_counter.update(tokens)
        # add candidate answers
        for a in question['choices']:
            if a not in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])

    # filter low-frequency words
    stopwords = set()
    for w, c in word_counter.items():
        if w and c >= min_cnt and w not in vocab['word_token_to_idx']:
            vocab['word_token_to_idx'][w] = len(vocab['word_token_to_idx'])
        if w and c >= stop_thresh:
            stopwords.add(w)

    # add candidate answers of val and test set
    for question in chain(val_set, test_set):
        for a in question['choices']:
            if a not in vocab['answer_token_to_idx']:
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])

    with open(output_path + '/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

def word2idx(vocab, word):
    result = [vocab['word_token_to_idx'].get(w, vocab['word_token_to_idx']['<UNK>'])
     for w in word_tokenize(word.lower())]
    return result

def gen_data(task, is_train):

    with open(output_path + '/entity_vocab.pkl', 'rb') as f:
        entity_vocab = pickle.load(f)

    with open(output_path + '/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    print('Convert word into idx')
    task_data = json.load(open(os.path.join(input_path, task + '.json')))
    question_ent, answer_choice, question, answer_label = [[] for _ in range(4)]
    if task == 'test':
        nlp = StanfordCoreNLP(r'/home/l/tool/stanford-corenlp-english-models', lang='en')
    for i in tqdm(range(len(task_data))):
    # for i in tqdm(range(1000)):
        ins = task_data[i]
        question.append(word2idx(vocab, ins['text']))
        answer_choice.append([word2idx(vocab, c) for c in ins['choices']])

        # tag entities in questions
        if task != 'test':
            spa = task_data[i]['sparql']
            ent = [e.lower() for e in re.findall(u'"(.*?)"', spa)]
        else:
            ner = nlp.ner(ins['text'])
            ent, e = [[] for _ in range(2)]
            for j in range(len(ner)):
                if ner[j][1] in ['O', 'NUMBER']:
                    if len(e) != 0:
                        ent.append(e)
                    e = []
                else:
                    e.append(ner[j][0].lower())
                if j == len(ner) - 1:
                    ent.append(e)
            ent = [' '.join(j) for j in ent]
        ent_lap = [entity_vocab[j] for j in ent if j in entity_vocab.keys()]
        question_ent.append(ent_lap)
        if task != 'test':
            answer_label.append([1 if ins['answer'] == c else 0 for c in ins['choices']])

    if is_train:
        return question, answer_choice, answer_label, question_ent
    else:
        return question, answer_choice, question_ent

def padding(item, max_len_1, max_len_2 = 0):
    if max_len_2 == 0:
        add_len = max_len_1 - len(item)
        item += [0] * add_len
    else:
        add_len = max_len_1 - len(item)
        item += [[0] * max_len_2] * add_len

        for i in range(len(item)):
            if len(item[i]) < max_len_2:
                add_len = max_len_2 - len(item[i])
                item[i] += [0] * add_len
            else:
                item[i] = item[i][: max_len_2]
    return item

class Mydata(Dataset):
    def __init__(self, task, is_train):
        super(Mydata, self).__init__()
        self.task = task
        self.is_train = is_train

        if self.is_train:
            self.question, self.answer_choice, self.answer_label, self.question_ent = gen_data(task, self.is_train)
        else:
            self.question, self.answer_choice, self.question_ent = gen_data(task, self.is_train)

    def __len__(self):
        return len(self.question)

    def __getitem__(self, item):
        if self.is_train:
            que = padding(self.question[item], 112)
            ans_c = padding(self.answer_choice[item], 10, 17)
            que_e = padding(self.question_ent[item], 4)
            ans_l = padding(self.answer_label[item], 10)

            que_item = torch.LongTensor(que)
            ans_c_item = torch.LongTensor(ans_c)
            que_e_item = torch.LongTensor(que_e)
            ans_l_item = torch.LongTensor(ans_l)

            return que_item, ans_c_item, ans_l_item, que_e_item

        else:
            que = padding(self.question[item], 112)
            ans_c = padding(self.answer_choice[item], 10, 17)
            que_e = padding(self.question_ent[item], 4)

            que_item = torch.LongTensor(que)
            ans_c_item = torch.LongTensor(ans_c)
            que_e_item = torch.LongTensor(que_e)

            return que_item, ans_c_item, que_e_item

def build_loader(batch_size, shuffle, drop_last, task, is_train):
    data_set = Mydata(task, is_train)
    data_iter = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                           drop_last=drop_last)
    return data_iter

if __name__ == '__main__':
    # prepare kg vocab
    entity_vocab, _, triples = load_kg(output_path)
    # prepare gcn input
    build_gcn(entity_vocab, triples)
    # prepare text vocab
    build_vocab()
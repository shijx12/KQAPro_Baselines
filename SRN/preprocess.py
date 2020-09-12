import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
import collections
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
from utils.misc import *
import re
import os


def encode_kb(args, vocab):
    kb = json.load(open(os.path.join(args.input_dir, 'kb.json')))
    concepts = kb['concepts']
    entities = kb['entities']
    adj_list = defaultdict(defaultdict)
    for concept in concepts:
        s = concept
        name = concepts[s]['name']
        # print(name)
        vocab['name2entity'][name] = s
        add_item_to_x2id(s, vocab['entity2id'])
        r = 'instanceOf'
        r_inv = r + '_inv'
        add_item_to_x2id(r, vocab['relation2id'])
        add_item_to_x2id(r_inv, vocab['relation2id']) 
        objs = concepts[concept]['instanceOf']            
        s_id, r_id, r_inv_id = vocab['entity2id'][s], vocab['relation2id'][r], vocab['relation2id'][r_inv]
        
        for o in objs:
            add_item_to_x2id(o, vocab['entity2id'])
            o_id = vocab['entity2id'][o]
            if not r_id in adj_list[s_id]:
                adj_list[s_id][r_id] = set()
            adj_list[s_id][r_id].add(o_id)
            if not r_inv_id in adj_list[o_id]:
                adj_list[o_id][r_inv_id] = set()
            adj_list[o_id][r_inv_id].add(s_id)
    
    for entity in entities:
        s = entity
        name = entities[s]['name']
        # print(name)
        vocab['name2entity'][name] = s
        add_item_to_x2id(s, vocab['entity2id'])
        r = 'instanceOf'
        r_inv = r + '_inv'
        add_item_to_x2id(r, vocab['relation2id'])
        add_item_to_x2id(r_inv, vocab['relation2id']) 
        objs = concepts[concept]['instanceOf']            
        s_id, r_id, r_inv_id = vocab['entity2id'][s], vocab['relation2id'][r], vocab['relation2id'][r_inv]
        
        for o in objs:
            add_item_to_x2id(o, vocab['entity2id'])
            o_id = vocab['entity2id'][o]
            if not r_id in adj_list[s_id]:
                adj_list[s_id][r_id] = set()
            adj_list[s_id][r_id].add(o_id)
            if not r_inv_id in adj_list[o_id]:
                adj_list[o_id][r_inv_id] = set()
            adj_list[o_id][r_inv_id].add(s_id)
        
        relations = entities[entity]['relations']
        for relation in relations:
            r = relation['predicate']
            r_inv = r + '_inv'
            add_item_to_x2id(r, vocab['relation2id'])
            add_item_to_x2id(r_inv, vocab['relation2id'])
            o = relation['object']
            add_item_to_x2id(o, vocab['entity2id'])
            r_id = vocab['relation2id'][r]
            r_inv_id = vocab['relation2id'][r_inv]
            dir = relation['direction']
            if dir == 'forward':
                s_id = vocab['entity2id'][s]
                o_id = vocab['entity2id'][o]
            else:
                s_id = vocab['entity2id'][o]
                o_id = vocab['entity2id'][s]
            if not r_id in adj_list[s_id]:
                adj_list[s_id][r_id] = set()
            adj_list[s_id][r_id].add(o_id)
            if not r_inv_id in adj_list[o_id]:
                adj_list[o_id][r_inv_id] = set()
            adj_list[o_id][r_inv_id].add(s_id)

    print('Save adj list')
    adj_list = dict(adj_list)

    # Sanity check
    print('Sanity check: {} entities'.format(len(vocab['entity2id'])))
    print('Sanity check: {} relations'.format(len(vocab['relation2id'])))
    num_facts = 0
    out_degrees = defaultdict(int)
    for s in adj_list:
        for r in adj_list[s]:
            num_facts += len(adj_list[s][r])
            out_degrees[s] += len(adj_list[s][r])
    print("Sanity check: maximum out degree: {}".format(max(out_degrees.values())))
    print('Sanity check: {} facts in knowledge graph'.format(num_facts))
    with open(os.path.join(args.output_dir, 'adj_list.pt'), 'wb') as f:
        pickle.dump(adj_list, f)



def encode_qa(args, vocab):
    def filter_qa_for_SRN(qa):
        answer = qa['answer']
        question = qa['rewrite']
        program = qa['program']
        if program[0]['function'] != 'Find':
            return None, None, None
        if program[-1]['function'] != 'What':
            return None, None, None
        e_s = None
        for input in program[0]['inputs']:
            if not input in vocab['name2entity']:
                continue
            if vocab['name2entity'][input] in vocab['entity2id']:
                if args.replace_es:
                    try:
                        question = question.replace(input, 'e_s')
                    except:
                        pass
                e_s = input
                break
        if e_s == None:
            return None, None, None
        if not answer in vocab['name2entity']:
            return None, None, None
        if answer in vocab['name2entity'] and vocab['name2entity'][answer] in vocab['entity2id']:
            return question, e_s, answer
        else:
            return None, None, None


    datasets = []
    for dataset in ['train', 'test', 'val']:
        data = json.load(open(os.path.join(args.input_dir, '%s.json'%(dataset))))
        filtered_data = []
        for qa in data:
            question, e_s, answer = filter_qa_for_SRN(qa)
            if question == None:
                continue
            filtered_data.append({'question': question, 'topic_entity': e_s, 'answer': answer})
        json.dump(filtered_data, open(os.path.join(args.output_dir, '%s.json'%(dataset)), 'w'))
        datasets.append(filtered_data)
    
    train_set, test_set, val_set = datasets[0], datasets[1], datasets[2]
    print('size of training data: {}'.format(len(train_set)))
    print('size of test data: {}'.format(len(test_set)))
    print('size of valid data: {}'.format(len(val_set)))
    print('Build question vocabulary')
    word_counter = Counter()
    for qa in tqdm(train_set):
        tokens = word_tokenize(qa['question'].lower())
        word_counter.update(tokens)
    # filter low-frequency words
    stopwords = set()
    for w, c in word_counter.items():
        if w and c >= args.min_cnt:
            add_item_to_x2id(w, vocab['word2id'])
        if w and c >= args.stop_thresh:
            stopwords.add(w)
    print('number of stop words (>={}): {}'.format(args.stop_thresh, len(stopwords)))
    print('number of word in dict: {}'.format(len(vocab['word2id'])))
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(vocab, dataset)
        print('shape of questions, topic_entities, answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)


def encode_dataset(vocab, dataset):
    questions = []
    topic_entities = []
    answers = []
    for qa in tqdm(dataset):
        assert len(qa['question']) > 0
        questions.append([vocab['word2id'].get(w, vocab['word2id']['<UNK>']) for w in word_tokenize(qa['question'].lower())])
        topic_entities.append([vocab['entity2id'][vocab['name2entity'][qa['topic_entity']]]])
        answers.append([vocab['entity2id'][vocab['name2entity'][qa['answer']]]])
    # question padding
    max_len = max(len(q) for q in questions)
    print('max question length:{}'.format(max_len))
    for q in questions:
        while len(q) < max_len:
            q.append(vocab['word2id']['<PAD>'])
    questions = np.asarray(questions, dtype=np.int32)
    topic_entities = np.asarray(topic_entities, dtype=np.int32)
    answers = np.asarray(answers, dtype=np.int32)
    return questions, topic_entities, answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '/data/csl/exp/AI_project/raw_data/MetaQA', type = str)
    parser.add_argument('--output_dir', default = '/data/csl/exp/AI_project/SRN/log', type = str)
    parser.add_argument('--min_cnt', type=int, default=1)
    parser.add_argument('--stop_thresh', type=int, default=1000)
    parser.add_argument('--num_hop', type = str, default = '1, 2, 3')
    parser.add_argument('--replace_es', action = 'store_true')
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print('Init vocabulary')
    vocab = {
        'word2id': init_word2id(),
        'entity2id': init_entity2id(),
        'relation2id': init_relation2id(),
        'name2entity': {}
    }

    print('Encode kb')
    encode_kb(args, vocab)

    print('Encode qa')
    encode_qa(args, vocab)

if __name__ == '__main__':
    main()


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

def write(f, s, r, o):
    s = str(s)
    r = str(r)
    o = str(o)
    # if o == 'yes' or o == 'no':
    #     print('\t'.join([s, r, o]))
    if o.lower() == 'yes' or o.lower() == 'no':
        return
    f.write('\t'.join([s, r, o]) + '\n')

    
def get_kbqa(args):
    kb = json.load(open(os.path.join(args.input_dir, 'kb.json')))
    Q2name = defaultdict(str)
    concepts = kb['concepts']
    entities = kb['entities']
    for concept in concepts:
        Q2name[concept] = concepts[concept]['name']
    for entity in entities:
        Q2name[entity] = entities[entity]['name']
    json.dump(Q2name, open(os.path.join(args.output_dir, 'Q2name.json'), 'w'))
    setting = args.setting
    with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
        for concept in concepts:
            s = Q2name[concept]
            r = 'instanceOf'
            objs = concepts[concept]['instanceOf']
            for o in objs:
                o = Q2name[o]
                write(f, s, r, o)
                # f.write('\t'.join([s, r, o]) + '\n')
        for entity in entities:
            s = Q2name[entity]
            r = 'instanceOf'
            objs = entities[entity]['instanceOf']
            for o in objs:
                o = Q2name[o]
                write(f, s, r, o)
                # f.write('\t'.join([s, r, o]) + '\n')
            relations = entities[entity]['relations']
            for relation in relations:
                direction = relation['direction']
                if direction == 'backward':
                    continue
                r = relation['predicate']
                o = relation['object']
                o = Q2name[o]
                write(f, s, r, o)
                # f.write('\t'.join([s, r, o]) + '\n')
                if 'qualifier' in setting:
                    qualifiers = relation['qualifiers']
                    for rq in qualifiers:
                        objs = qualifiers[rq]
                        for obj in objs:
                            oq = obj['value']
                            '''[rq, oq] for [s, r, o], construct virtual node'''
                            r_head_in = r + '_head_in'
                            r_tail_in = r + '_tail_in'
                            s_r_o = '_'.join([s, r, str(o)])
                            write(f, s, r_head_in, s_r_o)
                            write(f, o, r_tail_in, s_r_o)
                            write(f, s_r_o, rq, oq)
                            # f.write('\t'.join([s, r_head_in, s_r_o]) + '\n')
                            # f.write('\t'.join([o, r_tail_in, s_r_o]) + '\n')
                            # f.write('\t'.join([s_r_o, rq, oq]) + '\n')
            if not 'attr' in setting:
                continue
            attributes = entities[entity]['attributes']
            for attribute in attributes:
                r = attribute['key']
                o = attribute['value']['value']
                write(f, s, r, o)
                # f.write('\t'.join([s, r, o]) + '\n')
                if 'qualifier' in setting:
                    qualifiers = attribute['qualifiers']
                    for rq in qualifiers:
                        objs = qualifiers[rq]
                        for obj in objs:
                            oq = obj['value']
                            '''[rq, oq] for [s, r, o], construct virtual node'''
                            r_head_in = r + '_head_in'
                            r_tail_in = r + '_tail_in'
                            s_r_o = '_'.join([s, r, str(o)])
                            write(f, s, r_head_in, s_r_o)
                            write(f, o, r_tail_in, s_r_o)
                            write(f, s_r_o, rq, oq)
                            # f.write('\t'.join([s, r_head_in, s_r_o]) + '\n')
                            # f.write('\t'.join([o, r_tail_in, s_r_o]) + '\n')
                            # f.write('\t'.join([s_r_o, rq, oq]) + '\n')
    entities = defaultdict(int)
    relations = defaultdict(int)
    with open(os.path.join(args.output_dir, 'train.txt')) as f, open(os.path.join(args.output_dir, 'test.txt'), 'w') as f_test, open(os.path.join(args.output_dir, 'valid.txt'), 'w') as f_valid:
        train = f.readlines()
        for line in train:
            s, r, o = line.strip().split('\t')
            if not s in entities:
                entities[s] = len(entities)
            if not o in entities:
                entities[o] = len(entities)
            if not r in relations:
                relations[r] = len(relations)
        np.random.shuffle(train)
        test = train[:10000]
        valid = train[10000:20000]
        for line in test:
            f_test.write(line)
        for line in valid:
            f_valid.write(line)

    def find_topic_entity(program):
        for func in program:
            if func['function'] == 'Find':
                return func['inputs'][0]
        for func in program:
            if func['function'] == 'FilterConcept':
                return func['inputs'][0]
        return None

    def check(program, answer):
        if program[0]['function'] != 'Find':
            return False
        if program[-1]['function'] not in ['What', 'QueryAttr', 'QueryAttrUnderCondition', 'QueryAttrQualifier', 'QueryRelationQualifier']:
            return False
        if len(program[0]['inputs']) != 1:
            return False
        if program[0]['inputs'][0] not in entities:
            return False
        if answer not in entities:
            return False
        return True

    print(len(entities))
    print(len(relations))
    with open(os.path.join(args.output_dir, 'entitiy_ids.del'), 'w') as f:
        for entity in entities:
            f.write('\t'.join([str(entities[entity]), entity]) + '\n')
    
    with open(os.path.join(args.output_dir, 'relation_ids.del'), 'w') as f:
        for relation in relations:
            f.write('\t'.join([str(relations[relation]), relation]) + '\n')
    train = json.load(open(os.path.join(args.input_dir, 'train.json')))
    with open(os.path.join(args.output_dir, 'qa_train.txt'), 'w') as f:
        for item in tqdm(train):
            question = item['question']
            if '[' in question or ']' in question:
                question = question.replace('[', '').replace(']', '')
            answer = item['answer']
            program = item['program']
            if check(program, answer):
                topic_entity = program[0]['inputs'][0]
                if args.replace_es:
                    question = question.replace(topic_entity, 'e_s')
                f.write('\t'.join([question, topic_entity, answer]) + '\n')

        
    valid = json.load(open(os.path.join(args.input_dir, 'val.json')))
    with open(os.path.join(args.output_dir, 'qa_valid.txt'), 'w') as f:
        for item in tqdm(valid):
            question = item['question']
            if '[' in question or ']' in question:
                question = question.replace('[', '').replace(']', '')
            answer = item['answer']
            program = item['program']
            # if program[0]['function'] == 'Find' and len(program[0]['inputs']) == 1 and program[0]['inputs'][0] in entities and answer in entities:
            if check(program, answer):
                topic_entity = program[0]['inputs'][0]
                if args.replace_es:
                    question = question.replace(topic_entity, 'e_s')
                f.write('\t'.join([question, topic_entity, answer]) + '\n')


def encode_kbqa(args, vocab):
    adj_list = defaultdict(defaultdict)
    def add(s, r, o):
        add_item_to_x2id(s, vocab['entity2id'])
        add_item_to_x2id(o, vocab['entity2id'])
        add_item_to_x2id(r, vocab['relation2id'])
        r_inv = r + '_inv'
        add_item_to_x2id(r_inv, vocab['relation2id'])
        s_id, r_id, r_inv_id, o_id = vocab['entity2id'][s], vocab['relation2id'][r], vocab['relation2id'][r_inv], vocab['entity2id'][o]
        if not r_id in adj_list[s_id]:
            adj_list[s_id][r_id] = set()
        adj_list[s_id][r_id].add(o_id)
        if not r_inv_id in adj_list[o_id]:
            adj_list[o_id][r_inv_id] = set()
        adj_list[o_id][r_inv_id].add(s_id)

    with open(os.path.join(args.output_dir, 'train.txt')) as f:
        for line in f:
            s, r, o = line.strip().split('\t')
            add(s, r, o)
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
    
    datasets = []
    for dataset in ['train', 'valid']:
        with open(os.path.join(args.output_dir, 'qa_%s.txt'%(dataset)), 'r') as f:
            filtered_data = []
            for qa in f:
                question, e_s, answer = qa.strip().split('\t')
                filtered_data.append({'question': question, 'topic_entity': e_s, 'answer': answer})
            json.dump(filtered_data, open(os.path.join(args.output_dir, '%s.json'%(dataset)), 'w'))
            datasets.append(filtered_data)
    
    train_set, val_set = datasets[0], datasets[1]
    print('size of training data: {}'.format(len(train_set)))
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

    for name, dataset in zip(('train', 'val'), (train_set, val_set)):
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
        topic_entities.append([vocab['entity2id'][qa['topic_entity']]])
        answers.append([vocab['entity2id'][qa['answer']]])
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
    parser.add_argument('--input_dir', required = True, type = str)
    parser.add_argument('--output_dir', required = True, type = str)
    parser.add_argument('--setting', default = 'attr_qualifier', type = str)
    parser.add_argument('--min_cnt', type=int, default=1)
    parser.add_argument('--stop_thresh', type=int, default=1000)
    parser.add_argument('--replace_es', type = int, default = 1)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    get_kbqa(args)
    vocab = {
        'word2id': init_word2id(),
        'entity2id': init_entity2id(),
        'relation2id': init_relation2id()
    }
    encode_kbqa(args, vocab)
    
    
if __name__ == "__main__":
    main()

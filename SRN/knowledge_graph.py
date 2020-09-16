import collections
import os
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
from utils.misc import *

class KnowledgeGraph(nn.Module):
    def __init__(self, args, vocab):
        super(KnowledgeGraph, self).__init__()
        self.args = args
        self.entity2id, self.id2entity = vocab['entity2id'], vocab['id2entity']
        self.relation2id, self.id2relation = vocab['relation2id'], vocab['id2relation']
        self.adj_list = None
        self.action_space = None
        self.action_mask = None
        self.bandwidth = args.bandwidth
        with open(os.path.join(args.input_dir, 'adj_list.pt'), 'rb') as f:
            self.adj_list = pickle.load(f)
        self.vectorize_action_space()
        self.relation_embeddings = nn.Embedding(self.num_relations, args.dim_hidden)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    
    def vectorize_action_space(self):
        def load_pgrk_score():
            pgrk_scores = defaultdict(float)
            with open(os.path.join(self.args.input_dir, 'pgrk.txt')) as f:
                for line in f:
                    e, score = line.strip().split(':')
                    pgrk_scores[(int)(e)] = float(score)
            return pgrk_scores

        page_rank_scores = load_pgrk_score()

        def get_action_space(e1):
            action_space = []
            if e1 in self.adj_list:
                for r in self.adj_list[e1]:
                    targets = self.adj_list[e1][r]
                    for e2 in targets:
                        action_space.append((r, e2))
                if len(action_space) + 1 >= self.bandwidth:
                    # Base graph pruning
                    sorted_action_space = \
                        sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                    action_space = sorted_action_space[:self.bandwidth]
            action_space.insert(0, (NO_OP_RELATION_ID, e1))
            return action_space

        def vectorize_action_space(action_space_list, action_space_size):
            bucket_size = len(action_space_list)
            r_space = torch.zeros(bucket_size, action_space_size) + self.dummy_r
            e_space = torch.zeros(bucket_size, action_space_size) + self.dummy_e
            action_mask = torch.zeros(bucket_size, action_space_size)
            for i, action_space in enumerate(action_space_list):
                for j, (r, e) in enumerate(action_space):
                    r_space[i, j] = r
                    e_space[i, j] = e
                    action_mask[i, j] = 1
            return (r_space.long(), e_space.long()), action_mask

        self.action_space_buckets = {}
        action_space_buckets_discrete = defaultdict(list)
        self.entity2bucketid = torch.zeros(self.num_entities, 2).long()
        num_facts_saved_in_action_table = 0
        for e1 in range(self.num_entities):
            action_space = get_action_space(e1)
            key = int(len(action_space) / self.args.bucket_interval) + 1
            self.entity2bucketid[e1, 0] = key
            self.entity2bucketid[e1, 1] = len(action_space_buckets_discrete[key])
            action_space_buckets_discrete[key].append(action_space)
            num_facts_saved_in_action_table += len(action_space)
        print('Sanity check: {} facts saved in action table'.format(num_facts_saved_in_action_table - self.num_entities))
        for key in action_space_buckets_discrete:
            self.action_space_buckets[key] = vectorize_action_space(action_space_buckets_discrete[key], key * self.args.bucket_interval)
            print('Vectorize action spaces bucket {} with size {} finished'.format(key, len(self.action_space_buckets[key][-1])))
        print('Sanity check: {} action space bucket in total'.format(len(self.action_space_buckets)))


    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def self_edge(self):
        return NO_OP_RELATION_ID

    @property
    def self_e(self):
        return NO_OP_ENTITY_ID        

    @property
    def dummy_r(self):
        return DUMMY_RELATION_ID

    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID

    @property
    def dummy_start_r(self):
        return START_RELATION_ID

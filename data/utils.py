import json
from collections import Counter

def init_vocab():
    return {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }


"""
knowledge json format:
    'concepts':
    {
        'id':
        {
            'name': '',
            'instanceOf': ['<concept_id>'],
        }
    },
    'entities': # exclude concepts
    {
        'id': 
        {
            'name': '<entity_name>',
            'instanceOf': ['<concept_id>'],
            'attributes':
            [
                {
                    'key': '<key>',
                    'value': 
                    {
                        'type': 'string'/'quantity'/'date'/'year'
                        'value':  # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit':   # for quantity
                    },
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value, including keys type,value,unit
                        ]
                    }
                }
            ]
            'relations':
            [
                {
                    'predicate': '<predicate>',
                    'object': '<object_id>', # NOTE: it may be a concept id
                    'direction': 'forward' or 'backward',
                    'qualifiers':
                    {
                        '<qk>': 
                        [
                            <qv>, # each qv is a dictionary like value
                        ]
                    }
                }
            ]
        }
    }
"""
def get_kb_vocab(kb_json, min_cnt=1):
    counter = Counter()
    kb = json.load(open(kb_json))
    for i in kb['concepts']:
        counter.update([i, kb['concepts'][i]['name']])
    for i in kb['entities']:
        counter.update([i, kb['entities'][i]['name']])
        for attr_dict in kb['entities'][i]['attributes']:
            counter.update([attr_dict['key']])
            values = [attr_dict['value']]
            for qk, qvs in attr_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])+u])
        for rel_dict in kb['entities'][i]['relations']:
            counter.update([rel_dict['predicate'], rel_dict['direction']])
            values = []
            for qk, qvs in rel_dict['qualifiers'].items():
                counter.update([qk])
                values += qvs
            for value in values:
                u = value.get('unit', '')
                if u:
                    counter.update([u])
                counter.update([str(value['value'])+u])

    vocab = init_vocab()
    for v, c in counter.items():
        if v and c >= min_cnt and v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def invert_dict(d):
    return {v: k for k, v in d.items()}

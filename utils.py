from collections import defaultdict, Counter, deque
import torch
import json
import pickle
import numpy as np

def init_vocab():
    return {
        '<PAD>': 0,
        '<UNK>': 1,
        '<START>': 2,
        '<END>': 3
    }

def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_glove(glove_pt, idx_to_token):
    glove = pickle.load(open(glove_pt, 'rb'))
    dim = len(glove['the'])
    matrix = []
    for i in range(len(idx_to_token)):
        token = idx_to_token[i]
        tokens = token.split()
        if len(tokens) > 1:
            v = np.zeros((dim,))
            for token in tokens:
                v = v + glove.get(token, glove['the'])
            v = v / len(tokens)
        else:
            v = glove.get(token, glove['the'])
        matrix.append(v)
    matrix = np.asarray(matrix)
    return matrix


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

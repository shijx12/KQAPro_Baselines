import json
import pickle
import torch
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['word_idx_to_token'] = invert_dict(vocab['word_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    vocab['kb_idx_to_token'] = invert_dict(vocab['kb_token_to_idx'])
    vocab['predicate_idx_to_token'] = invert_dict(vocab['predicate_token_to_idx'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question, choices = list(map(torch.stack, batch[:2]))
    if batch[-1][0] is None:
        answer = None
    else:
        answer = torch.cat(batch[-1])
    return question, choices, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.questions, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            answer = None
        else:
            answer = torch.LongTensor([self.answers[index]])
        return question, choices, answer


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, kb_pt, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(3):
                inputs.append(pickle.load(f))
        with open(kb_pt, 'rb') as f:
            self.node_descs = torch.LongTensor(pickle.load(f))
            self.triples = torch.LongTensor(pickle.load(f))

        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab


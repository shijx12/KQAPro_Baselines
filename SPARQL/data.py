import json
import pickle
import torch
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['word_idx_to_token'] = invert_dict(vocab['word_token_to_idx'])
    vocab['sparql_idx_to_token'] = invert_dict(vocab['sparql_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question = torch.stack(batch[0])
    choices = torch.stack(batch[1])
    if batch[-1][0] is None:
        sparql, answer = None, None
    else:
        sparql = torch.stack(batch[2])
        answer = torch.cat(batch[3])
    return question, choices, sparql, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.questions, self.sparqls, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            sparql = None
            answer = None
        else:
            sparql = torch.LongTensor(self.sparqls[index])
            answer = torch.LongTensor([self.answers[index]])
        return question, choices, sparql, answer


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of word/sparql/answer: %d/%d/%d' % 
                (len(vocab['word_token_to_idx']), len(vocab['sparql_token_to_idx']), len(vocab['answer_token_to_idx'])))
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(4):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab


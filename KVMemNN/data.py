import json
import pickle
import torch
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['word_idx_to_token'] = invert_dict(vocab['word_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question, choices, keys, values = list(map(torch.stack, batch[:4]))
    if batch[-1][0] is None:
        answer = None
    else:
        answer = torch.cat(batch[-1])
    return question, choices, keys, values, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, all_keys, all_values, inputs):
        self.all_keys = all_keys
        self.all_values = all_values
        self.questions, self.key_indexes, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        key_index = self.key_indexes[index]
        keys = torch.LongTensor(self.all_keys[key_index])
        values = torch.LongTensor(self.all_values[key_index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            answer = None
        else:
            answer = torch.LongTensor([self.answers[index]])
        return question, choices, keys, values, answer


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, kb_pt, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(4):
                inputs.append(pickle.load(f))
        with open(kb_pt, 'rb') as f:
            all_keys = pickle.load(f)
            all_values = pickle.load(f)
        dataset = Dataset(all_keys, all_values, inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab


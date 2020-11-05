import json
import pickle
import torch
from utils.misc import invert_dict

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[-1][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    return source_ids, source_mask, choices, target_ids, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers = inputs
        self.is_test = len(self.answers)==0


    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        return source_ids, source_mask, choices, target_ids, answer


    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(5):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab
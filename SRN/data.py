import json
import pickle
import torch
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2word'] = invert_dict(vocab['word2id'])
    vocab['id2entity'] = invert_dict(vocab['entity2id'])
    vocab['id2relation'] = invert_dict(vocab['relation2id'])
    # vocab['entity2name'] = invert_dict(vocab['name2entity'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question, topic_entity, answer = list(map(torch.stack, batch))
    return question, topic_entity, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.questions, self.topic_entities, self.answers = inputs
        print(self.questions.shape)
        print(self.topic_entities.shape)
        print(self.answers.shape)

    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        topic_entity = torch.LongTensor(self.topic_entities[index])
        answer = torch.LongTensor(self.answers[index])
        return question, topic_entity, answer


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(3):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab


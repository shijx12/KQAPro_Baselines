import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm

from utils import MetricLogger, load_glove
from GRU.data import DataLoader
from GRU.model import GRUClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed


def validate(args, vocab, model, data, device):
    def write(f, predict):
        predict = predict.squeeze().tolist()
        for i in predict:
            f.write(vocab['answer_idx_to_token'][i] + '\n')
    model.eval()
    count, correct = 0, 0
    f1 = open(os.path.join(args.save_dir, 'predict.txt'), 'w')
    f2 = open(os.path.join(args.save_dir, 'choiced_predict.txt'), 'w')
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            question, choices, program, prog_depends, prog_inputs, sparql, answer = [x.to(device) for x in batch]
            logit = model(question)
            predict = logit.max(1)[1]
            write(f1, predict)
            choiced_logit = torch.gather(logit, 1,choices) # [bsz, num_choices]
            choiced_predict = torch.gather(choices, 1, choiced_logit.max(1)[1].unsqueeze(-1)) # [bsz, 1]
            write(f2, choiced_predict)
            correct += torch.eq(predict, answer).long().sum().item()
            count += len(answer)
    f1.close()
    f2.close()
    acc = correct / count


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    test_loader = DataLoader(vocab_json, test_pt, args.batch_size)
    vocab = train_loader.vocab

    model = GRUClassifier(vocab, args.dim_word, args.dim_hidden)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)




    validate(args, vocab, model, test_loader, device)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--glove_pt', default='/data/csl/resources/word2vec/glove.840B.300d.py36.pt')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--ckpt', required = True)
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()

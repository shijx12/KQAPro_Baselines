import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
import pickle
import argparse
import shutil

from utils import MetricLogger, load_glove
from RGCN.data import DataLoader
from RGCN.model import QuesAnsByRGCN

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed


def validate(model, data, device):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            question, choices, answer = [x.to(device) for x in batch]
            logit = model(question)
            predict = logit.max(1)[1]
            correct += torch.eq(predict, answer).long().sum().item()
            count += len(answer)

    acc = correct / count
    logging.info('\nValid Accuracy: %.4f\n' % acc)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    kb_pt = os.path.join(args.input_dir, 'kb.pt')
    train_loader = DataLoader(vocab_json, kb_pt, train_pt, args.batch_size, training=True)
    train_loader_large_bsz = DataLoader(vocab_json, kb_pt, train_pt, 128, training=True)
    val_loader = DataLoader(vocab_json, kb_pt, val_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    node_descs = train_loader.node_descs.to(device)
    node_descs = node_descs[:, :args.max_desc]
    triples = train_loader.triples.to(device)
    triples = triples[:args.max_triple]
    model = QuesAnsByRGCN(vocab, 
        node_descs, triples, 
        args.dim_word, args.dim_hidden, args.dim_g)
    logging.info("Load pretrained word vectors.........")
    pretrained = load_glove(args.glove_pt, vocab['word_idx_to_token'])
    with torch.no_grad():
        model.input_embeddings.weight.set_(torch.Tensor(pretrained))
    model = model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.lr_decay_step], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    # validate(model, val_loader, device)
    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")
    for epoch in range(args.num_epoch):
        model.train()
        if epoch < 2:
            _train_loader = train_loader_large_bsz
            only_q = True
        else:
            _train_loader = train_loader
            only_q = False
        for iteration, batch in enumerate(_train_loader):
            iteration = iteration + 1

            question, choices, answer = [x.to(device) for x in batch]
            logits = model(question, only_q)
            loss = criterion(logits, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meters.update(loss=loss.item())

            if iteration % (len(train_loader) // 1000) == 0:
                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        
        validate(model, val_loader, device)
        scheduler.step()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_{}.pt'.format(epoch)))



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', default='/data/sjx/glove.840B.300d.py36.pt')

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_decay_step', default=5, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=512, type=int)
    parser.add_argument('--dim_g', default=32, type=int)
    parser.add_argument('--max_desc', default=20, type=int)
    parser.add_argument('--max_triple', default=200000, type=int)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    # set random seed
    torch.manual_seed(args.seed)

    train(args)

if __name__ == '__main__':
    main()


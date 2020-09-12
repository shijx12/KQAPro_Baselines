import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm

from utils.misc import MetricLogger, load_glove
from .data import DataLoader
from .model import KVMemNN

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1) # avoid using multiple cpus

def validate(model, data, device):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            question, choices, keys, values, answer = [x.to(device) for x in batch]
            logit = model(question, keys, values)
            predict = logit.max(1)[1]
            correct += torch.eq(predict, answer).long().sum().item()
            count += len(answer)

    acc = correct / count
    logging.info('\nValid Accuracy: %.4f\n' % acc)
    return acc


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    kb_pt = os.path.join(args.input_dir, 'kb.pt')
    train_loader = DataLoader(vocab_json, kb_pt, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, kb_pt, val_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    model = KVMemNN(
            args.num_hop, 
            args.dim_emb, 
            vocab
            )
    logging.info("Load pretrained word vectors.........")
    pretrained = load_glove(args.glove_pt, vocab['word_idx_to_token'])
    with torch.no_grad():
        model.embeddings.weight.set_(torch.Tensor(pretrained))
    model = model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 50], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    validate(model, val_loader, device)
    meters = MetricLogger(delimiter="  ")
    best_acc = 0
    logging.info("Start training........")
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, choices, keys, values, answer = [x.to(device) for x in batch]
            logits = model(question, keys, values)
            loss = criterion(logits, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meters.update(loss=loss.item())

            if iteration % (len(train_loader) // 100) == 0:
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
        
        acc = validate(model, val_loader, device)
        scheduler.step()
        if acc and acc > best_acc:
            best_acc = acc
            logging.info("\nupdate best ckpt with acc: {:.4f}".format(best_acc))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', required=True)

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_hop', default=3, type=int)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    # set random seed
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()

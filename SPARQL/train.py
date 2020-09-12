import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date

from utils.misc import MetricLogger
from utils.load_kb import DataForSPARQL
from .data import DataLoader
from .model import SPARQLParser
from .sparql_engine import get_sparql_answer
from .preprocess import postprocess_sparql_tokens

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

def whether_equal(answer, pred):
    """
    check whether the two arguments are equal as attribute value
    """
    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = '{} {}'.format(str(v), ' '.join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split('-')
            y_split = y.split('-')
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def validate(args, kb, model, data, device):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            question, choices, sparql, answer = [x.to(device) for x in batch]
            pred_sparql = model(question)

            answer, pred_sparql = [x.cpu().numpy().tolist() for x in (answer, pred_sparql)]
            for a, s in zip(answer, pred_sparql):
                given_answer = data.vocab['answer_idx_to_token'][a]
                s = [data.vocab['sparql_idx_to_token'][i] for i in s]
                end_idx = len(s)
                if '<END>' in s:
                    end_idx = s.index('<END>')
                s = ' '.join(s[1:end_idx])
                s = postprocess_sparql_tokens(s)
                pred_answer = get_sparql_answer(s, kb)
                is_match = whether_equal(given_answer, pred_answer)
                if is_match:
                    correct += 1
            count += len(answer)
    acc = correct / count
    logging.info('\nValid Accuracy: %.4f\n' % acc)
    return acc

def test_sparql(args):
    # check whether the SPARQL engine is correct, with the training set
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    data = DataLoader(vocab_json, train_pt, args.batch_size, training=False)
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))

    count, correct = 0, 0
    for batch in tqdm(data, total=len(data)):
        question, choices, sparql, answer = batch
        pred_sparql = sparql

        answer = answer.cpu().numpy().tolist()
        pred_sparql = pred_sparql.cpu().numpy().tolist()
        for a, s in zip(answer, pred_sparql):
            given_answer = data.vocab['answer_idx_to_token'][a]
            s = [data.vocab['sparql_idx_to_token'][i] for i in s]
            end_idx = len(s)
            if '<END>' in s:
                end_idx = s.index('<END>')
            s = ' '.join(s[1:end_idx])
            s = postprocess_sparql_tokens(s)
            pred_answer = get_sparql_answer(s, kb)
            is_match = whether_equal(given_answer, pred_answer)
            count += 1
            if is_match:
                correct += 1
            else:
                print(given_answer, pred_answer)

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))

    logging.info("Create model.........")
    model = SPARQLParser(vocab, args.dim_word, args.dim_hidden, args.max_dec_len)
    model = model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 50], gamma=0.1)

    # validate(args, kb, model, val_loader, device)
    meters = MetricLogger(delimiter="  ")
    best_acc = 0
    logging.info("Start training........")
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, choices, sparql, answer = [x.to(device) for x in batch]
            loss = model(question, sparql)
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
        
        acc = validate(args, kb, model, val_loader, device)
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

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--max_dec_len', default=100, type=int)
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
        logging.info(k+':'+str(v))

    # set random seed
    torch.manual_seed(args.seed)

    train(args)
    # test_sparql(args)


if __name__ == '__main__':
    main()

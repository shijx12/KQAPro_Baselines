import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm

from utils.misc import MetricLogger
from SRN.data import DataLoader
from SRN.model import SRN

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1) # avoid using multiple cpus

def validate(args, vocab, model, data, device):
    def write(f, predict):
        predict = predict.squeeze().tolist()
        for i in predict:
            f.write(vocab['id2entity'][i] + '\n')
    model.eval()
    count, correct = 0, 0
    f1 = open(os.path.join(args.save_dir, 'predict.txt'), 'w')
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, topic_entities, answers = [x.to(device) for x in batch]
            predict = model(questions, topic_entities)
            
            pred_e2s = predict['pred_e2s']
            pred_e2_scores = predict['pred_e2_scores']
            search_traces = predict['search_traces']
            pred_top_e2 = pred_e2s[:, 0].unsqueeze(-1) # [bsz, beam_size] => [bsz] => [bsz, 1]
            write(f1, pred_top_e2)
            correct += torch.any(pred_top_e2 == answers, dim=1).float().sum().item()
            count += len(answers)
    acc = correct / count
    f1.close()
    logging.info('\nValid Accuracy: %.4f\n' % acc)
    return acc

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

    model = SRN(args, args.dim_word, args.dim_hidden, vocab)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    validate(args, vocab, model, test_loader, device)




def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', default='/data/csl/resources/word2vec/glove.840B.300d.py36.pt')

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_rollout_steps', default=3, type=int)
    parser.add_argument('--num_rollouts', default=10, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=300, type=int)
    parser.add_argument('--bucket_interval', default = 3, type = int)
    parser.add_argument('--opt', default = 'adam', type = str)
    parser.add_argument('--bandwidth', default = 100, type = int)
    parser.add_argument('--gamma', default = 0.95, type = float)
    parser.add_argument('--eta', default = 0.95, type = float)
    parser.add_argument('--beta', default = 0, type =float)
    parser.add_argument('--beam_size', default = 32, type = int)
    parser.add_argument('--log_name', default = 'log.txt', type = str)
    parser.add_argument('--model_name', default = 'model.pt', type = str)
    parser.add_argument('--rel', action = 'store_true')
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()

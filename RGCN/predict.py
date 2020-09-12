import os
import torch
import argparse
import json
from tqdm import tqdm

from .data import DataLoader
from .model import QuesAnsByRGCN


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('load test data')
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    kb_pt = os.path.join(args.input_dir, 'kb.pt')
    data = DataLoader(vocab_json, kb_pt, test_pt, 4)
    vocab = data.vocab

    print('load model')
    node_descs = data.node_descs.to(device)
    node_descs = node_descs[:, :args.max_desc]
    triples = data.triples.to(device)
    triples = triples[:args.max_triple]
    model = QuesAnsByRGCN(vocab, 
        node_descs, triples, 
        args.dim_word, args.dim_hidden, args.dim_g)
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
    
    fn_open = open(os.path.join(args.save_dir, 'predict.txt'), 'w')
    fn_choice = open(os.path.join(args.save_dir, 'choice_predict.txt'), 'w')
    for batch in tqdm(data, total=len(data)):
        question, choices, answer = batch
        question = question.to(device)
        logit = model(question)
        logit = logit.detach().cpu()

        for l, c in zip(logit, choices):
            a = l.max(0)[1].item()
            a = vocab['answer_idx_to_token'][a]
            fn_open.write(a + '\n')
            # mask for multi-choice
            l = torch.softmax(l, 0)
            mask = torch.ones((len(l),)).bool()
            mask[c] = 0
            l[mask] = 0
            a = l.max(0)[1].item()
            a = vocab['answer_idx_to_token'][a]
            fn_choice.write(a + '\n')
    fn_open.close()
    fn_choice.close()


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to store predictions')

    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=512, type=int)
    parser.add_argument('--dim_g', default=32, type=int)
    parser.add_argument('--max_desc', default=20, type=int)
    parser.add_argument('--max_triple', default=200000, type=int)
    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()

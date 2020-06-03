import os
import torch
import argparse
import json
from tqdm import tqdm

from load_kb import DataForSPARQL
from .data import DataLoader
from .model import SPARQLParser
from .sparql_engine import get_sparql_answer
from .preprocess import remove_space

import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('load test data')
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    data = DataLoader(vocab_json, test_pt, 128, training=False)
    vocab = data.vocab
    kb = DataForSPARQL(args.kb_path)
    raw_data = json.load(open(args.test_program_json))

    print('load model')
    model = SPARQLParser(vocab, args.dim_word, args.dim_hidden, args.max_dec_len)
    model = model.to(device)
    model.load_state_dict(torch.load(args.ckpt))
    
    f = open(args.output_fn, 'w')
    idx = 0
    for batch in tqdm(data, total=len(data)):
        question, choices, sparql, answer = batch
        question = question.to(device)
        pred_sparql = model(question)

        pred_sparql = pred_sparql.cpu().numpy().tolist()
        for s in pred_sparql:
            last_func = raw_data[idx]['program'][-1] 
            # TODO: predict the last function  instead of load from gt
            # TODO: support date==year in evaluation
            idx += 1
            s = [vocab['sparql_idx_to_token'][i] for i in s]
            end_idx = len(s)
            if '<END>' in s:
                end_idx = s.index('<END>')
            s = ' '.join(s[1:end_idx])
            s = remove_space(s)
            answer = str(get_sparql_answer(s, last_func, kb))
            f.write(answer + '\n')
    f.close()



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--ckpt', required=True, help='path of checkpoints')
    parser.add_argument('--output_fn', required=True, help='path to store predictions')
    parser.add_argument('--kb_path', required=True)
    parser.add_argument('--test_program_json', help='visit the program of test set, should be replaced by a predictor')

    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--max_dec_len', default=100, type=int)
    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()

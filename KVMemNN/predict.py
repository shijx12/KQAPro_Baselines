import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm

from .data import DataLoader
from .model import KVMemNN


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path of checkpoint')

    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_hop', default=3, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    kb_pt = os.path.join(args.input_dir, 'kb.pt')
    test_loader = DataLoader(vocab_json, kb_pt, test_pt, 32)
    vocab = test_loader.vocab


    model = KVMemNN(
            args.num_hop, 
            args.dim_emb, 
            vocab
            )
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
    model = model.to(device)
    model.eval()
    
    def write(f, predict):
        predict = predict.squeeze().tolist()
        for i in predict:
            f.write(vocab['answer_idx_to_token'][i] + '\n')

    f1 = open(os.path.join(args.save_dir, 'predict.txt'), 'w')
    f2 = open(os.path.join(args.save_dir, 'choice_predict.txt'), 'w')
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            question, choices, keys, values = [x.to(device) for x in batch[:4]]
            logit = model(question, keys, values) # [bsz, num_answers]
            predict = logit.max(1)[1]
            write(f1, predict)
            choiced_logit = torch.gather(logit, 1,choices) # [bsz, num_choices]
            choiced_predict = torch.gather(choices, 1, choiced_logit.max(1)[1].unsqueeze(-1)) # [bsz, 1]
            write(f2, choiced_predict)
    f1.close()
    f2.close()


if __name__ == '__main__':
    main()

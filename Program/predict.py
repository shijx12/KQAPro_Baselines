import os
import torch
import argparse
import shutil
from tqdm import tqdm
import numpy as np

from .data import DataLoader
from .parser import Parser
from .executor_rule import RuleExecutor

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path of checkpoint')
    # model hyperparameters
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    test_loader = DataLoader(vocab_json, test_pt, 128)
    vocab = test_loader.vocab

    rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))
    model = Parser(vocab, args.dim_word, args.dim_hidden)

    print("load ckpt from {}".format(args.save_dir))
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, 'model.pt'), map_location={'cuda': 'cpu'}))
    model = model.to(device)
    model.eval() 

    with open(os.path.join(args.save_dir, 'predict.txt'), 'w') as f:
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                question, choices = [x.to(device) for x in batch[:2]]
                pred_program, pred_inputs = model(question)

                pred_program, pred_inputs = [x.cpu().numpy() for x in (pred_program, pred_inputs)]
                for i in range(len(pred_program)):
                    pred = rule_executor.forward(pred_program[i], pred_inputs[i], ignore_error=True)
                    f.write(str(pred) + '\n')
    print("save predictions into {}".format(os.path.join(args.save_dir, 'predict.txt')))

if __name__ == '__main__':
    main()

import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np

from utils import MetricLogger, load_glove
from Program.data import DataLoader
from Program.executor_rule import RuleExecutor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed


def validate_executor(executor, data):
    correct = 0
    count = 0
    for batch in tqdm(data, total=len(data)):
        question, choices, gt_program, gt_dep, gt_inputs, answer = batch
        gt_program, gt_dep, gt_inputs = [x.cpu().numpy() for x in (gt_program, gt_dep, gt_inputs)]
        answer = [data.vocab['answer_idx_to_token'][a.item()] for a in answer]
        preds = []
        for i in range(len(gt_program)):
            pred = executor.forward(gt_program[i], gt_dep[i], gt_inputs[i], ignore_error=True)
            if pred == answer[i]:
                correct += 1
            else:
                print(pred, answer[i])
                pred = executor.forward(gt_program[i], gt_dep[i], gt_inputs[i], ignore_error=True, show_details=True)
                embed()
            count += 1
    print('{}/{}/{:.4f}'.format(correct, count, correct/count))


def validate(model, data, device, executor=None):
    model.eval()
    end_id = data.vocab['function_token_to_idx']['<END>']
    match_prog_num = 0
    match_dep_num = 0
    match_inp_num = 0
    match_all_num = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            question, choices, gt_program, gt_dep, gt_inputs, answer = [x.to(device) for x in batch]
            pred_program, pred_dep, pred_inputs = model(question)

            gt_program, gt_dep, gt_inputs = [x.cpu().numpy() for x in (gt_program, gt_dep, gt_inputs)]
            pred_program, pred_dep, pred_inputs = [x.cpu().numpy() for x in (pred_program, pred_dep, pred_inputs)]

            for i in range(len(gt_program)):

                # print(gt_program[i])
                # print(gt_dep[i])
                # print(gt_inputs[i])
                # print('---')
                # print(pred_program[i])
                # print(pred_dep[i])
                # print(pred_inputs[i])
                # print('==========')

                match = True
                for j in range(min(len(gt_program[i]), len(pred_program[i]))):
                    if gt_program[i, j] != pred_program[i, j]:
                        match = False
                        break
                    if gt_program[i, j] == end_id and pred_program[i, j] == end_id:
                        l = j
                        break
                if match:
                    match_prog_num += 1
                    if np.all(gt_dep[i,1:l,:]==pred_dep[i,1:l,:]):
                        match_dep_num += 1
                    if np.all(gt_inputs[i,1:l,:]==pred_inputs[i,1:l,:]):
                        match_inp_num += 1
                    if np.all(gt_dep[i,1:l,:]==pred_dep[i,1:l,:]) and \
                        np.all(gt_inputs[i,1:l,:]==pred_inputs[i,1:l,:]):
                        match_all_num += 1

            count += len(gt_program)

            if executor:
                answer = [data.vocab['answer_idx_to_token'][a.item()] for a in answer]
                for i in range(len(gt_program)):
                    pred = executor.forward(pred_program[i], pred_dep[i], pred_inputs[i], ignore_error=True)
                    if pred == answer[i]:
                        correct += 1

    logging.info('\nValid match program: {:.4f}, dependencies: {:.4f}, inputs: {:.4f}, all: {:.4f}\n'.format(
        match_prog_num / count,
        match_dep_num / count,
        match_inp_num / count,
        match_all_num / count
        ))
    if executor:
        logging.info('Accuracy: {:.4f}\n'.format(correct / count))


def train(args):
    if args.sequential_input:
        from Program.sequential_input.parser import Parser
    else:
        from Program.token_input.parser import Parser

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    vocab = train_loader.vocab

    rule_executor = RuleExecutor(vocab, args.kb_json, args.sequential_input)

    logging.info("Create model.........")
    model = Parser(vocab, args.dim_word, args.dim_hidden)
    logging.info("Load pretrained word vectors.........")
    pretrained = load_glove(args.glove_pt, vocab['word_idx_to_token'])
    with torch.no_grad():
        model.word_embeddings.weight.set_(torch.Tensor(pretrained))
    model = model.to(device)
    logging.info(model)
    if args.ckpt and os.path.exists(args.ckpt):
        logging.info("load ckpt from {}".format(args.ckpt))
        model.load_state_dict(torch.load(args.ckpt, map_location={'cuda': 'cpu'}))

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[args.lr_decay_step], gamma=0.1)

    validate(model, val_loader, device)
    # validate_executor(rule_executor, train_loader)

    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, choices, program, prog_depends, prog_inputs, answer = [x.to(device) for x in batch]
            loss = model(question, program, prog_depends, prog_inputs)
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
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
        scheduler.step()
        if epoch == args.num_epoch-1 or (epoch+1)%5 == 0:
            validate(model, val_loader, device, rule_executor)
        else:
            validate(model, val_loader, device)



def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', default='/data/sjx/glove.840B.300d.py36.pt')
    parser.add_argument('--kb_json')
    parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--lr_decay_step', default=2, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--sequential_input', action='store_true')
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
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


if __name__ == '__main__':
    main()

import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date
from utils.misc import MetricLogger, seed_everything, ProgressBar
from utils.load_kb import DataForSPARQL
from .data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from .sparql_engine import get_sparql_answer
import torch.optim as optim
import logging
import time
from utils.lr_scheduler import get_linear_schedule_with_warmup
import re
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
# def post_process(text):
#     pattern = re.compile(r'".*?"')
#     named_entities = pattern.findall(text)
#     # mask = '"' + '#' * 10 + '"'
#     # text = pattern.sub(mask, text)
#     for idx, named_entity in enumerate(named_entities):
#         mask = '"' + str(idx) + '#' * 10 + str(idx) + '"'
#         text = pattern.sub(mask, text, 1)
#     print(text)
#     text = text.replace('?', ' ?').replace('.', ' .')
#     for idx, named_entity in enumerate(named_entities):
#         pattern = '"' + str(idx) + '#' * 10 + str(idx) + '"'
#         mask = named_entity
#         text = text.replace(pattern, mask)
#         # text = pattern.sub(named_entity, text, 1)
#     print(text)
#     return text

def post_process(text):
    pattern = re.compile(r'".*?"')
    nes = []
    for item in pattern.finditer(text):
        nes.append((item.group(), item.span()))
    pos = [0]
    for name, span in nes:
        pos += [span[0], span[1]]
    pos.append(len(text))
    assert len(pos) % 2 == 0
    assert len(pos) / 2 == len(nes) + 1
    chunks = [text[pos[i]: pos[i+1]] for i in range(0, len(pos), 2)]
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
    bingo = ''
    for i in range(len(chunks) - 1):
        bingo += chunks[i] + nes[i][0]
    bingo += chunks[-1]
    return bingo

def vis(args, kb, model, data, device, tokenizer):
    while True:
        # text = 'Who is the father of Tony?'
        # text = 'Donald Trump married Tony, where is the place?'
        text = input('Input your question:')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text], max_length = 512, pad_to_max_length = True, return_tensors="pt", truncation = True)
            source_ids = input_ids['input_ids'].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )
            outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in outputs]
            outputs = [post_process(output) for output in outputs]
            print(outputs[0])

def validate(args, kb, model, data, device, tokenizer):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        all_answers = []
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, choices, target_ids, answer = [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_answers.extend(answer.cpu().numpy())
            # break
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        print(outputs)
        pred_sparql = [post_process(output) for output in outputs]
        given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
        # print(outputs)
        res = []
        for a, s in tqdm(zip(given_answer, pred_sparql)):
            pred_answer = get_sparql_answer(s, kb)
            print(s)
            print(pred_answer)
            print(a)
            res.append({'pred_sparql': s, 'pred_ans': pred_answer, 'gold_ans': a, 'is_correct': pred_sparql == a})
            is_match = whether_equal(a, pred_answer)
            if is_match:
                correct += 1
            count += 1
        json.dump(res, open('pred_sparql.json', 'w'))
    acc = correct / count
    logging.info('\nValid Accuracy: %.4f\n' % acc)
    return acc 

def predict(args, kb, model, data, device, tokenizer):
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        for batch in tqdm(data, total=len(data)):
            batch = batch[:3]
            source_ids, source_mask, choices = [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids=source_ids,
                max_length = 500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            # break
        outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
        pred_sparql = [post_process(output) for output in outputs]
        with open(os.path.join(args.save_dir, 'predict.txt'), 'w') as f:
            for sparql in tqdm(pred_sparql):
                pred_answer = get_sparql_answer(sparql, kb)

                if pred_answer == None:
                    pred_answer = 'None'
                f.write(pred_answer + '\n')

    #     for a, s in tqdm(zip(given_answer, pred_sparql)):
    #         pred_answer = get_sparql_answer(s, kb)
    #         print(s)
    #         print(pred_answer)
    #         print(a)
    #         is_match = whether_equal(a, pred_answer)
    #         if is_match:
    #             correct += 1
    #         count += 1
    # acc = correct / count
    # logging.info('\nValid Accuracy: %.4f\n' % acc)
    # return acc 


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
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.ckpt)
    model = model_class.from_pretrained(args.ckpt)
    model = model.to(device)
    logging.info(model)
    # predict(args, kb, model, val_loader, device, tokenizer)

    validate(args, kb, model, val_loader, device, tokenizer)
    # vis(args, kb, model, val_loader, device, tokenizer)
def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', required=True)

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    
    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default = 1e-4, type = float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()


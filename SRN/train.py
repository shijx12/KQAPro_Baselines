import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm

from utils.misc import MetricLogger, load_glove
from SRN.data import DataLoader
from SRN.model import SRN
import copy
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
            questions, topic_entities, answers = [x.to(device) for x in batch]
            predict = model(questions, topic_entities)
            pred_e2s = predict['pred_e2s']
            pred_e2_scores = predict['pred_e2_scores']
            search_traces = predict['search_traces']
            pred_top_e2 = pred_e2s[:, 0].unsqueeze(-1) # [bsz, beam_size] => [bsz] => [bsz, 1]
            correct += torch.any(pred_top_e2 == answers, dim=1).float().sum().item()
            count += len(answers)
    acc = correct / count
    logging.info('\nValid Accuracy: %.4f' % acc)
    return acc

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader, val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    model = SRN(args, args.dim_word, args.dim_hidden, vocab)
    logging.info("Load pretrained word vectors.........")
    pretrained = load_glove(args.glove_pt, vocab['id2word'])
    model.word_embeddings.weight.data = torch.Tensor(pretrained)
    model = model.to(device)
    logging.info(model)
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3], gamma=0.1)

    validate(model, val_loader, device)
    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")
    best_model= copy.deepcopy(model.state_dict())
    best_acc = 0.0
    eps = 0.00001
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, topic_entity, answer = [x.to(device) for x in batch]
            loss, pt_loss = model(question, topic_entity, answer)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meters.update(loss=pt_loss.item())

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
            break

        
        acc = validate(model, val_loader, device)
        if acc > best_acc + eps:
            best_acc = acc
            no_update = 0
            best_model = copy.deepcopy(model.state_dict())
            logging.info("Validation accuracy increased from previous epoch {}".format(acc))
            torch.save(model.state_dict(), os.path.join(args.save_dir, '%s-%s-%s-%s.pt'%(args.opt, str(args.lr), str(args.bandwidth), str(epoch))))
        elif (acc < best_acc + eps) and (no_update < args.patience):
            no_update +=1
            logging.info("Validation accuracy decreases to %f from %f, %d more epoch to check"%(acc, best_acc, args.patience-no_update))
        elif no_update == args.patience:
            logging.info("Model has exceed patience. Saving best model and exiting")
            torch.save(best_model, os.path.join(args.save_dir, "best_score_model.pt"))
            exit()

        # acc = validate(model, test_loader, device)
        # torch.save(model.state_dict(), os.path.join(args.save_dir, '%s-%s-%d-%.2f'%(args.model_name, args.opt, args.lr, acc)))
        # scheduler.step()


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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_rollout_steps', default=3, type=int)
    parser.add_argument('--num_rollouts', default=10, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=300, type=int)
    parser.add_argument('--bucket_interval', default = 3, type = int)
    parser.add_argument('--opt', default = 'adam', type = str)
    parser.add_argument('--bandwidth', default = 50, type = int)
    parser.add_argument('--gamma', default = 0.95, type = float)
    parser.add_argument('--eta', default = 0.95, type = float)
    parser.add_argument('--beta', default = 0, type =float)
    parser.add_argument('--beam_size', default = 32, type = int)
    parser.add_argument('--log_name', default = 'log.txt', type = str)
    parser.add_argument('--model_name', default = 'model.pt', type = str)
    parser.add_argument('--patience', default = 10, type = int)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
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

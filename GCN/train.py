import torch
import torch.optim as optim
import numpy as np
import random
from GCN.preprocess import build_loader, golve_attr
from GCN.model import gcn_qa_model
from tqdm import tqdm
from time import time
import os
import pickle

current_path = os.path.dirname(__file__)
previous_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
input_path = previous_path + '/test_dataset'
output_path = previous_path + '/preprocess'

# avoid randomly train
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def train(model_path, model_name):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    train_data = build_loader(batch_size=128, shuffle=True, drop_last=True, task='train')
    val_data = build_loader(batch_size=128, shuffle=True, drop_last=True, task='val')
    with open(output_path + '/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    emb_size = 300
    channel = [[300, 150], [150, 300]]
    model = gcn_qa_model(emb_size=emb_size, input_size=len(vocab['word_token_to_idx']), channel=channel)

    print('Loading pretrained word embedding')
    with open(output_path + '/glove.840B.300d.py36.pt', 'rb') as f:
        glove = pickle.load(f)
    vocab_pretrained = [golve_attr(k, glove) for k in vocab['word_token_to_idx']]
    with torch.no_grad():
        model.word_embedding.weight.set_(torch.Tensor(vocab_pretrained))
    # model = model.to(device)
    model.cuda()

    time_s = time()
    best_loss = 9999
    early_stop = 2
    state = {}

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

    for e in range(100):
        train_loss, train_n = [0 for _ in range(2)]
        print('training... e: %d' % (e))
        model.train()
        for batch in tqdm(train_data):
            if torch.cuda.is_available():
                batch = [b.cuda() for b in batch]
            optimizer.zero_grad()
            loss = model(batch, is_train=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_n += 1

        val_loss, val_n = [0 for _ in range(2)]
        print('evaluating...')
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_data):
                batch = [b.cuda() for b in batch]
                loss = model(batch, is_train=True)

                val_loss += loss.item()
                val_n += 1
        print('time: %d' % (time() - time_s))
        print('train_loss: %.4f, val_loss: %.4f' % (train_loss / train_n, val_loss / val_n))

        if val_loss/ val_n < best_loss:
            best_loss = val_loss/ val_n
            early_stop = 0
            state['model_state'] = model.state_dict()
            state['loss'] = best_loss
            state['e'] = e
            state['time'] = time() - time_s
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            torch.save(state, model_path + '/' + model_name + '.pkl')
        else:
            early_stop += 1
            if early_stop > 2:
                break
# def main():
#     current_path = os.getcwd()
#     previous_path = os.path.abspath(os.path.dirname(os.getcwd()))
#
#     parser = argparse.ArgumentParser()
#     # input and output
#     # parser.add_argument('--input_dir', required=True)
#     # parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
#     parser.add_argument('--input_dir', default=previous_path + '/preprocess/')
#     parser.add_argument('--save_dir', default=previous_path + 'result/')
#     # parser.add_argument('--glove_pt', default='/home/l/task/KBQA_Baselines-master/preprocess/glove.840B.300d.py36.pt')
#     parser.add_argument('--glove_pt', default=previous_path + '/preprocess/glove.840B.300d.py36.pt')
#
#     # training parameters
#     parser.add_argument('--lr', default=0.001, type=float)
#     parser.add_argument('--weight_decay', default=1e-5, type=float)
#     parser.add_argument('--num_epoch', default=10, type=int)
#     parser.add_argument('--batch_size', default=16, type=int)
#     parser.add_argument('--seed', type=int, default=666, help='random seed')
#     # model hyperparameters
#     parser.add_argument('--dim_emb', default=300, type=int)
#     parser.add_argument('--num_hop', default=3, type=int)
#     args = parser.parse_args()
#
#     # make logging.info display into both shell and file
#     if os.path.isdir(args.save_dir):
#         shutil.rmtree(args.save_dir)
#     os.mkdir(args.save_dir)
#     fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)
#     # args display
#     for k, v in vars(args).items():
#         logging.info(k + ':' + str(v))
#
#     # set random seed
#     torch.manual_seed(args.seed)
#
#     train(args)

if __name__ == '__main__':
    model_name = 'gcn_model'
    model_path = previous_path + '/model'
    train(model_path, model_name)

import torch
import numpy as np
from GCN.preprocess import build_loader, gen_data
from GCN.model import gcn_qa_model
from tqdm import tqdm
import os
import pickle

current_path = os.path.dirname(__file__)
previous_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
input_path = previous_path + '/test_dataset'
output_path = previous_path + '/preprocess'

model_path = previous_path + '/model'
model_name = 'gcn_model'

state = torch.load(model_path + '/' + model_name + '.pkl')
print(model_name)
print('loss: %.4f, e: %d, time: %d' % (state['loss'], state['e'], state['time']))

val_data = build_loader(batch_size=128, shuffle=False, drop_last=False, task='val', is_train=False)
with open(output_path + '/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
emb_size = 300
channel = [[300, 150], [150, 300]]
model = gcn_qa_model(emb_size=emb_size, input_size=len(vocab['word_token_to_idx']), channel=channel)
model.cuda()

answer_pro = []
with torch.no_grad():
    print('testing...')
    model.eval()
    for batch in tqdm(val_data):
        batch = [b.cuda() for b in batch]
        ans_pre= model(batch, is_train=False)
        answer_pro += ans_pre.cpu().tolist()
answer_pre = [np.argsort(line)[-1] for line in answer_pro]
print(answer_pre)
with open(output_path + '/answer_predicate.pkl', 'wb') as f:
    pickle.dump(answer_pre, f)

# with open(output_path + '/answer_predicate.pkl', 'rb') as f:
#     answer_pre = pickle.load(f)

_, _, answer_label, _ = gen_data('val', is_train = True)
answer_gold = [j for line in answer_label for j in range(len(line)) if line[j] == 1]
bingo = (sum([1 for i in range(len(answer_pre)) if answer_pre[i] == answer_gold[i]]))
acc = bingo/ len(answer_pre)
print(acc)
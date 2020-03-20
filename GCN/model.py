import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geometric
from models.BiGRU import BiGRU
import os
import pickle
from time import time

current_path = os.getcwd()
previous_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    """

    def __init__(self, in_channel, out_channel):
        super(GraphConv, self).__init__()

        self.conv1 = geometric.nn.SAGEConv(in_channel[0], out_channel[0])
        self.conv2 = geometric.nn.SAGEConv(in_channel[1], out_channel[1])

    def forward(self, x, edge_indices, edge_weight):
        x = self.conv1(x, edge_indices, edge_weight)
        x = F.leaky_relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices, edge_weight)
        x = F.dropout(x)
        x = F.normalize(x)

        return x

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x

class gcn_qa_model(nn.Module):
    def __init__(self, emb_size, input_size, channel):
        super(gcn_qa_model, self).__init__()

        # word_embedding
        self.word_embedding = nn.Embedding(input_size, emb_size, padding_idx=0)

        # encoder
        self.encoder = BiGRU(emb_size, emb_size, num_layers=2, dropout=0.2)

        # gcn
        self.gcn = GraphConv(channel[0], channel[1])

        # answer_fc
        self.answer_fc = Linear(emb_size, 1)

    def forward(self, batch, is_train):
        if is_train:
            question, answer_choice, answer_label, question_ent = batch
        else:
            question, answer_choice, question_ent = batch

        choice = answer_choice.view(-1, answer_choice.size(-1))

        # encode question
        question_vec = self.word_embedding(question)
        question_vec_d = F.dropout(question_vec, 0.3)
        question_len = question.ne(0).sum(dim=1)
        _, question_emb, _ = self.encoder(question_vec_d, question_len)
        # question_emb, _, _ = self.encoder(question_vec_d, question_len)
        # question_emb_new = question_emb.unsqueeze(1).expand(question_emb.size(0), answer_choice.size(1),
        #                                                     question_emb.size(1), -1)
        # question_emb_new = question_emb_new.contiguous().view(question.size(0), -1, question_emb.size(-1))

        # encode answer_choice
        choice_vec = self.word_embedding(choice)
        choice_vec_d = F.dropout(choice_vec, 0.3)
        choice_len = choice.ne(0).sum(dim=1)
        _, choice_emb, _ = self.encoder(choice_vec_d, choice_len)
        choice_emb = choice_emb.view(question.size(0), answer_choice.size(1), -1)

        # gcn
        with open(previous_path + '/preprocess/gcn_input.pkl', 'rb') as f:
            gcn_input = pickle.load(f)
        entity_attr = torch.FloatTensor(gcn_input['entity_attr']).cuda()
        edge_index = torch.LongTensor(gcn_input['edge_index']).cuda().transpose(1, 0)
        edge_attr = torch.LongTensor(gcn_input['edge_attr']).cuda()
        gcn_embedding = self.gcn(entity_attr, edge_index, edge_attr)

        # kg-que att
        question_net_len = question_ent.ne(0).sum(dim=1)
        question_emb_n = []
        for i in range(question.size(0)):
            ent_e = [gcn_embedding[j] for j in range(question_net_len[i])]
            if len(ent_e) != 0:
                ent_emb = torch.stack(ent_e).cuda()
                a = torch.mm(question_emb[i].view(1, -1), ent_emb.transpose(1, 0))
                s = torch.softmax(a, -1)
                que_n = torch.mm(s, ent_emb)
                question_emb_n.append(que_n.view(-1))
            else:
                question_emb_n.append(question_emb[i])
        question_emb_new = torch.stack(question_emb_n).cuda()

        # que-choice att
        question_emb_new = question_emb_new.unsqueeze(1)
        a = torch.bmm(choice_emb, question_emb_new.transpose(1, -1)).view(question.size(0), -1)
        s = torch.softmax(a, -1).contiguous().view(question.size(0), -1, 1)
        choice_emb_new = torch.bmm(s, question_emb_new)

        if is_train:
            # answer prediction
            answer_pre = self.answer_fc(choice_emb_new).view(-1)
            loss = F.binary_cross_entropy_with_logits(answer_pre, answer_label.float().view(-1))
            return loss
        else:
            # answer prediction
            answer_pre = self.answer_fc(choice_emb_new).view(question.size(0), -1)
            return answer_pre

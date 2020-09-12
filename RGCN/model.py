"""
Refer to https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv

from utils.BiGRU import BiGRU

class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=2, dropout=0,
                 use_self_loop=False, use_cuda=True):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return None
        # return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
        #         self.num_bases, activation=None,
        #         self_loop=self.use_self_loop)

    def forward(self, g, h, r, norm=None):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class QuesAnsByRGCN(nn.Module):
    def __init__(self, vocab, node_descs, edge_triples,
            dim_word, dim_hidden, dim_g, num_bases=1, num_hidden_layers=1):
        """
        Args:
            - edge_triples (np.array) [#triple, 3]
        """
        super().__init__()
        num_rels = len(vocab['predicate_token_to_idx'])
        num_desc_word = len(vocab['kb_token_to_idx'])
        num_question_word = len(vocab['word_token_to_idx'])
        num_class = len(vocab['answer_token_to_idx'])

        self.rgcn = RGCN(dim_g, dim_g, dim_g, num_rels, num_bases, num_hidden_layers)
        edge_src  = edge_triples[:,0]
        edge_type = edge_triples[:,1]
        edge_dst = edge_triples[:,2]
        self.edge_type = edge_type
        self.num_nodes = len(node_descs)
        self.node_descs = node_descs # [#node, max_desc]
        self.dim_g = dim_g

        self.desc_embeddings = nn.Embedding(num_desc_word, dim_g)
        nn.init.normal_(self.desc_embeddings.weight, mean=0, std=1/math.sqrt(dim_g))

        self.input_embeddings = nn.Embedding(num_question_word, dim_word)
        nn.init.normal_(self.input_embeddings.weight, mean=0, std=1/math.sqrt(dim_word))

        self.word_dropout = nn.Dropout(0.3)
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.0)

        # create graph
        self.g = DGLGraph()
        self.g.add_nodes(self.num_nodes)
        self.g.add_edges(edge_src, edge_dst)

        self.lin_h_to_g = nn.Linear(dim_hidden, dim_g)
        self.classifier = nn.Sequential(
                nn.Linear(dim_g + dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_class)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, questions, only_q=False):
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        question_input = self.word_dropout(self.input_embeddings(questions))
        _, question_embeddings, _ = self.question_encoder(question_input, question_lens)
        # [bsz, dim_h]

        if only_q:
            bsz = question_embeddings.size(0)
            device = question_embeddings.device
            empty = torch.zeros((bsz, self.dim_g)).to(device)
            feat = torch.cat((empty, question_embeddings), dim=1)
            logits = self.classifier(feat)
            return logits


        agg_feats = []
        bsz = len(questions)
        for i in range(bsz):
            # construct initial node features
            q = question_embeddings[i].view(1, 1, -1) # [1, 1, dim_h]
            node_desc_emb = self.word_dropout(self.desc_embeddings(self.node_descs))
            # [#node, max_desc, dim_g]
            q_g = self.lin_h_to_g(q) # [1, 1, dim_g]
            attn = torch.softmax(torch.sum(node_desc_emb * q_g, dim=2), dim=1) # [#node, max_desc]
            node_feat = torch.sum(attn.unsqueeze(2) * node_desc_emb, dim=1) # [#node, dim_g]

            # rgcn
            node_feat = self.rgcn(self.g, node_feat, self.edge_type) # [#node, dim_g]

            # answer feature
            q_g = q_g.view(1, -1) # [1, dim_g]
            attn = torch.softmax(torch.sum(node_feat * q_g, dim=1, keepdim=True), dim=0) # [#node, 1]
            node_agg = torch.sum(node_feat * attn, dim=0) # [dim_g]
            node_agg = torch.cat((node_agg, q.view(-1)), dim=0) # [dim_g+dim_h]
            agg_feats.append(node_agg)
        
        agg_feats = torch.stack(agg_feats) # [bsz, 2*dim_h]
        logits = self.classifier(agg_feats)
        return logits

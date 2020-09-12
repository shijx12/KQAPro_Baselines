import torch
import torch.nn as nn

from utils.BiGRU import BiGRU, GRU

class KVMemNN(nn.Module):
    def __init__(self, num_hop, dim_emb, vocab):
        super().__init__()
        self.num_hop = num_hop
        num_vocab = len(vocab['word_token_to_idx'])
        num_class = len(vocab['answer_token_to_idx'])
        
        self.embeddings = nn.Embedding(num_vocab, dim_emb)
        self.question_encoder = BiGRU(dim_emb, dim_emb, num_layers=2, dropout=0.2)
        self.word_dropout = nn.Dropout(0.3)
        self.linears = []
        for i in range(num_hop):
            lin = nn.Linear(dim_emb, dim_emb)
            self.linears.append(lin)
            self.add_module('linear_{}'.format(i), lin)

        self.classifier = nn.Sequential(
                nn.Linear(dim_emb, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_class)
            )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, questions, keys, values):
        """
        Args:
            questions [bsz, max_q_len]
            keys [bsz, num_slot, max_k_len]
            values [bsz, num_slot, max_v_len]
        """
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.embeddings(questions))
        q, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens)
        q = self.embeddings(questions).sum(dim=1) # [bsz, dim_emb]
        k = self.embeddings(keys).sum(dim=2) # [bsz, num_slot, dim_emb]
        v = self.embeddings(values).sum(dim=2) # [bsz, num_slot, dim_emb]

        for i in range(self.num_hop):
            weights = torch.bmm(k, q.unsqueeze(2)).squeeze(2) # [bsz, num_slot]
            weights = torch.softmax(weights, dim=1)
            o = torch.bmm(weights.unsqueeze(1), v).squeeze(1) # [bsz, dim_emb]
            q = self.linears[i](q + o) # [bsz, dim_emb]     
        logits = self.classifier(q)
        return logits

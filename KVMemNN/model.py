import torch
import torch.nn as nn

from models.BiGRU import BiGRU

class KVMemNN(nn.Module):
    def __init__(self, num_hop, dim_emb, vocab):
        super().__init__()
        self.num_hop = num_hop
        num_vocab = len(vocab['word_token_to_idx'])
        num_class = len(vocab['answer_token_to_idx'])
        
        self.embeddings = nn.Embedding(num_vocab, dim_emb)
        
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

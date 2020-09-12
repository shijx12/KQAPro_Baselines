import torch
import torch.nn as nn

from utils.BiGRU import GRU, BiGRU

class SPARQLParser(nn.Module):
    def __init__(self, vocab, dim_word, dim_hidden, max_dec_len):
        super().__init__()
        num_words = len(vocab['word_token_to_idx'])
        num_sparql = len(vocab['sparql_token_to_idx'])
        self.vocab = vocab
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.max_dec_len = max_dec_len

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.3)
        self.question_encoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.sparql_embeddings = nn.Embedding(num_sparql, dim_word)
        self.decoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.sparql_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_sparql),
            )

        self.att_lin = nn.Linear(dim_hidden, dim_hidden)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, questions, sparqls=None):
        """
        Args:
            questions [bsz, max_q]
            sparqls [bsz, max_s]
        Return:
            if sparqls are given, then return losses
            else, return predicted sparqls
        """
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions))
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens)
        # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]

        if sparqls is None: # during inference
            return self.inference(q_word_h, q_embeddings, q_hn)
        else:
            return self.train_phase(q_word_h, q_embeddings, q_hn, sparqls)


    def train_phase(self, q_word_h, q_embeddings, q_hn, sparqls):
        bsz, max_s = sparqls.size(0), sparqls.size(1)
        device = sparqls.device
        sparql_lens = max_s - sparqls.eq(0).long().sum(dim=1) # 0 means <PAD>
        sparql_mask = sparqls.ne(0).long()

        s_word_emb = self.word_dropout(self.sparql_embeddings(sparqls))
        s_word_h, _, _ = self.decoder(s_word_emb, sparql_lens, h_0=q_hn) # [bsz, max_s, dim_h]
        # attention over question words
        attn = torch.softmax(torch.bmm(s_word_h, q_word_h.permute(0, 2, 1)), dim=2) # [bsz, max_s, max_q]
        attn_word_h = torch.bmm(attn, q_word_h) # [bsz, max_s, dim_h]
        # sum up
        s_word_h = s_word_h + attn_word_h # [bsz, max_s, dim_h]

        criterion = nn.CrossEntropyLoss().to(device)
        logit = self.sparql_classifier(s_word_h) # [bsz, max_s, num_sparql]
        loss = criterion(logit.permute(0, 2, 1)[:,:,:-1], sparqls[:,1:]) # remember to shift the gt

        return loss


    def inference(self, q_word_h, q_embeddings, q_hn):
        """
        Predict sparqls
        """
        bsz = q_word_h.size(0)
        device = q_word_h.device
        start_id = self.vocab['sparql_token_to_idx']['<START>']
        end_id = self.vocab['sparql_token_to_idx']['<END>']

        latest_sparql = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
        last_h = q_hn
        finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced

        # store predictions at each step
        sparqls = [latest_sparql]

        for i in range(self.max_dec_len):
            s_word_emb = self.word_dropout(self.sparql_embeddings(latest_sparql)).unsqueeze(1) # [bsz, 1, dim_w]
            s_word_h, last_h = self.decoder.forward_one_step(s_word_emb, last_h) # [bsz, 1, dim_h]
            # attention over question words
            attn = torch.softmax(torch.bmm(s_word_h, q_word_h.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
            attn_word_h = torch.bmm(attn, q_word_h) # [bsz, 1, dim_h]
            # sum up
            s_word_h = s_word_h + attn_word_h # [bsz, 1, dim_h]

            logit = self.sparql_classifier(s_word_h).squeeze(1) # [bsz, num_sparql]
            latest_sparql = torch.argmax(logit, dim=1) # [bsz, ]
            sparqls.append(latest_sparql)

            finished = finished | latest_sparql.eq(end_id).byte()
            if finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break

        sparqls = torch.stack(sparqls, dim=1) # [bsz, max_s]

        return sparqls

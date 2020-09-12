import torch
import torch.nn as nn

from utils.BiGRU import GRU, BiGRU

class Parser(nn.Module):
    def __init__(self, vocab, dim_word, dim_hidden, max_dec_len=20, max_inp=3):
        super().__init__()
        num_func = len(vocab['function_token_to_idx'])
        num_words = len(vocab['word_token_to_idx'])
        self.vocab = vocab
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.max_dec_len = max_dec_len
        self.max_inp = max_inp

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)
        self.question_encoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.func_embeddings = nn.Embedding(num_func, dim_word)
        self.decoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.func_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_func),
            )

        self.inp_embeddings = nn.Embedding(num_words, dim_word)
        self.inp_decoder = GRU(dim_word + dim_hidden, dim_hidden, num_layers=2, dropout=0.2)
        self.inp_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_words),
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, questions, programs=None, inputs=None):
        """
        Args:
            questions [bsz, max_q]
            programs [bsz, max_prog]
            inputs [bsz, max_prog, max_inp=3]
        Return:
            if programs are given, then return losses
            else, return predicted programs
        """
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions))
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens)
        # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]

        if programs is None: # during inference
            return self.inference(q_word_h, q_embeddings, q_hn)
        else:
            return self.train_phase(q_word_h, q_embeddings, q_hn, programs, inputs)


    def train_phase(self, q_word_h, q_embeddings, q_hn, programs, inputs):
        bsz, max_prog = programs.size(0), programs.size(1)
        device = programs.device
        program_lens = programs.size(1) - programs.eq(0).long().sum(dim=1) # 0 means <PAD>
        program_mask = programs.ne(0).long()

        p_word_emb = self.word_dropout(self.func_embeddings(programs))
        p_word_h, _, _ = self.decoder(p_word_emb, program_lens, h_0=q_hn) # [bsz, max_prog, dim_h]
        # attention over question words
        attn = torch.softmax(torch.bmm(p_word_h, q_word_h.permute(0, 2, 1)), dim=2) # [bsz, max_prog, max_q]
        attn_word_h = torch.bmm(attn, q_word_h) # [bsz, max_prog, dim_h]
        # sum up
        p_word_h = p_word_h + attn_word_h # [bsz, max_prog, dim_h]


        criterion_CE = nn.CrossEntropyLoss().to(device)
        # predict function
        logit_func = self.func_classifier(p_word_h) # [bsz, max_prog, num_func]
        loss_func = criterion_CE(logit_func.permute(0, 2, 1)[:,:,:-1], programs[:,1:]) # remember to shift the gt

        # remove inputs of function <START>
        inputs = inputs[:,1:,:].view(bsz, -1) # [bsz, (max_prog-1)*3]
        # add an extra <START> at the beginning, for convenience of inference
        start_token = torch.zeros((bsz, 1)).to(device).fill_(self.vocab['word_token_to_idx']['<START>']).long()
        inputs = torch.cat((start_token, inputs), dim=1) # [bsz, 1+(max_prog-1)*3]
        inp_emb = self.word_dropout(self.inp_embeddings(inputs)) # [bsz, 1+(max_prog-1)*3, dim_w]
        
        rep_p_word_h = p_word_h.view(bsz, max_prog, 1, -1).expand(-1, -1, 3, -1).\
                reshape(bsz, max_prog*3, -1).contiguous() # [bsz, max_prog*3, dim_h]
        # align, so that func <START> is used to predict the 3 inputs of the first function
        rep_p_word_h = rep_p_word_h[:, :1+(max_prog-1)*3]
        inp_h, _, _ = self.inp_decoder(torch.cat((inp_emb, rep_p_word_h), dim=2), 
                1+(program_lens-1)*3, h_0=q_hn) # [bsz, 1+(max_prog-1)*3, dim_h]
        # attention over question words
        attn = torch.softmax(torch.bmm(inp_h, q_word_h.permute(0, 2, 1)), dim=2)
        attn_word_h = torch.bmm(attn, q_word_h)
        # sum up
        inp_h = inp_h + attn_word_h # [bsz, 1+(max_prog-1)*3, dim_h]
        # logit
        logit_inp = self.inp_classifier(inp_h) # [bsz, 1+(max_prog-1)*3, dim_h]
        loss_inp = criterion_CE(logit_inp.permute(0, 2, 1)[:,:,:-1], inputs[:,1:]) # shift the input <START>

        loss = loss_func + loss_inp

        return loss


    def inference(self, q_word_h, q_embeddings, q_hn):
        """
        Predict programs, and inputs
        """
        bsz = q_word_h.size(0)
        device = q_word_h.device
        start_id = self.vocab['function_token_to_idx']['<START>']
        end_id = self.vocab['function_token_to_idx']['<END>']

        latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
        last_h = q_hn
        finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced

        latest_inp = torch.LongTensor([self.vocab['word_token_to_idx']['<START>']]*bsz).to(device) # [bsz, ]
        last_inp_h = q_hn

        # store predictions at each step
        programs = [latest_func]
        inputs = [torch.zeros((bsz, self.max_inp)).long().to(device)]

        for i in range(self.max_dec_len):
            p_word_emb = self.word_dropout(self.func_embeddings(latest_func)).unsqueeze(1) # [bsz, 1, dim_w]
            p_word_h, last_h = self.decoder.forward_one_step(p_word_emb, last_h) # [bsz, 1, dim_h]
            # attention over question words
            attn = torch.softmax(torch.bmm(p_word_h, q_word_h.permute(0, 2, 1)), dim=2) # [bsz, 1, max_q]
            attn_word_h = torch.bmm(attn, q_word_h) # [bsz, 1, dim_h]
            # sum up
            p_word_h = p_word_h + attn_word_h # [bsz, 1, dim_h]

            # predict function
            logit_func = self.func_classifier(p_word_h).squeeze(1) # [bsz, num_func]
            latest_func = torch.argmax(logit_func, dim=1) # [bsz, ]
            programs.append(latest_func)

            # predict input
            pred_inp = []
            for _ in range(self.max_inp):
                inp_emb = self.word_dropout(self.inp_embeddings(latest_inp)).unsqueeze(1) # [bsz, 1, dim_w]
                inp_h, last_inp_h = self.inp_decoder.forward_one_step(
                        torch.cat((inp_emb, p_word_h), dim=2), 
                        last_inp_h) # [bsz, 1, dim_h]
                attn = torch.softmax(torch.bmm(inp_h, q_word_h.permute(0, 2, 1)), dim=2)
                attn_word_h = torch.bmm(attn, q_word_h)
                inp_h = inp_h + attn_word_h # [bsz, 1, dim_h]

                logit_inp = self.inp_classifier(inp_h).squeeze(1) # [bsz, num_word]
                latest_inp = torch.argmax(logit_inp, dim=1) # [bsz, ]
                pred_inp.append(latest_inp)
            pred_inp = torch.stack(pred_inp, dim=1) # [bsz, 3]
            inputs.append(pred_inp)

            finished = finished | latest_func.eq(end_id).byte()
            if finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break

        programs = torch.stack(programs, dim=1) # [bsz, max_prog]
        inputs = torch.stack(inputs, dim=1) # [bsz, max_prog, 3]
        return programs, inputs


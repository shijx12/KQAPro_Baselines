import torch
import torch.nn as nn

from models.BiGRU import GRU, BiGRU

class Parser(nn.Module):
    def __init__(self, vocab, dim_word, dim_hidden, max_dec_len=20, max_dep=2, max_inp=3, max_inp_len=12):
        super().__init__()
        num_func = len(vocab['function_token_to_idx'])
        num_words = len(vocab['word_token_to_idx'])
        self.vocab = vocab
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.max_dec_len = max_dec_len
        self.max_inp = max_inp
        self.max_inp_len = max_inp_len

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.3)
        self.question_encoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.func_embeddings = nn.Embedding(num_func, dim_word)
        self.decoder = GRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)

        self.func_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_func),
            )
        self.dep_lin = nn.Linear(dim_hidden, dim_hidden)
        self.dep_classifier = nn.Sequential(
                nn.Linear(dim_hidden*2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
            )
        self.inp_lin = nn.Linear(dim_hidden, dim_hidden)
        self.inp_decoders = []
        self.inp_classifiers = []
        for i in range(max_inp):
            d = GRU(dim_word, dim_hidden, num_layers=1, dropout=0.0)
            self.inp_decoders.append(d)
            self.add_module('inp_decoders_{}'.format(i), d)
            m = nn.Linear(dim_hidden, num_words)
            self.inp_classifiers.append(m)
            self.add_module('inp_classifiers_{}'.format(i), m)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, questions, programs=None, dependencies=None, inputs=None):
        """
        Args:
            questions [bsz, max_q]
            programs [bsz, max_prog]
            dependencies [bsz, max_prog, max_dep=2]
            inputs [bsz, max_prog, max_inp=3, max_inp_len=12]
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
            return self.train_phase(q_word_h, q_embeddings, q_hn, programs, dependencies, inputs)


    def train_phase(self, q_word_h, q_embeddings, q_hn, programs, dependencies, inputs):
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
        criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        # predict function
        logit_func = self.func_classifier(p_word_h) # [bsz, max_prog, num_func]
        loss_func = criterion_CE(logit_func.permute(0, 2, 1)[:,:,:-1], programs[:,1:]) # remember to shift the gt
        # predict dependencies by pairwise comparison
        loss_dep = []
        dep_word_h = self.dep_lin(p_word_h)
        for i in range(1, programs.size(1)-1): # predict for step i+1
            candidates = dep_word_h[:, :i] # [bsz, i, dim_h], all functions before the i-th
            current = dep_word_h[:, i:i+1] # [bsz, 1, dim_h]
            inter = torch.cat([candidates, current.expand_as(candidates)], dim=2) # [bsz, i, 2*dim_h]
            logit = self.dep_classifier(inter).squeeze(2) # [bsz, i]
            gt_index = dependencies[:,i+1] # [bsz, 2], should be the dependencies of next step
            gt = torch.zeros((bsz, i+1)).to(device)
            gt.scatter_(dim=1, index=gt_index, value=1)
            gt = gt[:,1:].float() # shift the gt, and remove padding at the same time
            # skip training of <PAD> function
            mask = program_mask[:,i+1]
            if mask.sum().item() == 0:
                break
            loss_dep.append(criterion_BCE(logit[mask], gt[mask]))
        loss_dep = sum(loss_dep) / len(loss_dep)

        # predict inputs with multiple classifiers
        loss_inp = []
        inp_h0 = self.inp_lin(p_word_h) # [bsz, max_prog, dim_h], initial hidden state for input decoder
        inp_word_emb = self.word_dropout(self.word_embeddings(inputs)) # [bsz, max_prog, max_inp, max_inp_len, dim_w]
        for i in range(self.max_inp):
            decoder = self.inp_decoders[i]
            classifier = self.inp_classifiers[i]

            # pack implement
            pack_inp_word_emb = inp_word_emb[:,:-1,i].reshape(bsz*(max_prog-1), self.max_inp_len, self.dim_word) 
            # [bsz*(max_prog-1), max_inp_len, dim_w], decoding input
            pack_inp = inputs[:,1:,i].reshape(bsz*(max_prog-1), self.max_inp_len) 
            # [bsz*(max_prog-1), max_inp_len], decoding target, should be right shift
            pack_inp_len = self.max_inp_len - pack_inp.eq(0).long().sum(dim=1)
            pack_inp_h0 = inp_h0[:,:-1].reshape(1, bsz*(max_prog-1), self.dim_hidden)
            # [1, bsz*(max_prog-1), dim_h]
            pack_inp_word_h, _, _ = decoder(pack_inp_word_emb, pack_inp_len, h_0=pack_inp_h0)
            # [bsz*(max_prog-1), max_inp_len, dim_h]
            logit_inp = classifier(pack_inp_word_h) # [bsz*(max_prog-1), max_inp_len, num_word]
            loss_inp.append(
                criterion_CE(logit_inp.permute(0, 2, 1)[:,:,:-1], pack_inp[:, 1:])
                )

            # step-by-step implement
            # for step in range(max_prog-1):
            #     # prepare what decoder needs for certain step
            #     step_inp_word_emb = inp_word_emb[:,step,i] # [bsz, max_inp_len, dim_w]
            #     step_inp = inputs[:,step+1,i] # [bsz, max_inp_len], NOTE the target should be next step
            #     step_inp_len = self.max_inp_len - step_inp.eq(0).long().sum(dim=1) # 0 means <PAD>
            #     step_inp_h0 = inp_h0[:, step].unsqueeze(0) # [1, bsz, dim_h]
            #     # decode
            #     step_inp_word_h, _, _ = decoder(step_inp_word_emb, step_inp_len, h_0=step_inp_h0) 
            #     # [bsz, max_inp_len, dim_h]
            #     logit_inp = classifier(step_inp_word_h) # [bsz, max_inp_len, num_word]
            #     loss_inp.append(criterion_CE(logit_inp.permute(0, 2, 1)[:,:,:-1], step_inp[:, 1:]))
        loss_inp = sum(loss_inp) / len(loss_inp)

        loss = loss_func + loss_dep + loss_inp
        return loss


    def inference(self, q_word_h, q_embeddings, q_hn):
        """
        Predict programs, dependencies, and inputs
        """
        bsz = q_word_h.size(0)
        device = q_word_h.device
        start_id = self.vocab['function_token_to_idx']['<START>']
        end_id = self.vocab['function_token_to_idx']['<END>']

        latest_func = torch.LongTensor([start_id]*bsz).to(device) # [bsz, ]
        last_h = q_hn
        finished = torch.zeros((bsz,)).byte().to(device) # record whether <END> is produced

        # store predictions at each step
        history_ph = []
        programs = [latest_func]
        dependencies = [torch.zeros((bsz, 2)).long().to(device)]
        inputs = [torch.zeros((bsz, self.max_inp, self.max_inp_len)).long().to(device)]

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

            # predict dependencies
            if i == 0:
                latest_dep = torch.zeros((bsz, 2)).long().to(device)
            else:
                candidates = torch.cat(history_ph[:i], dim=1) # [bsz, i, dim_h], all functions before the i-th
                candidates = self.dep_lin(candidates)
                dep_word_h = self.dep_lin(p_word_h) # [bsz, 1, dim_h]
                inter = torch.cat([candidates, dep_word_h.expand_as(candidates)], dim=2) # [bsz, i, 2*dim_h]
                prob = torch.sigmoid(self.dep_classifier(inter).squeeze(2)) # [bsz, i]
                if i == 1:
                    indices = prob.gt(0.5).long() # 0 for padding
                    indices = torch.cat([indices, torch.zeros((bsz, 1)).long().to(device)], dim=1) # [bsz, 2]
                else:
                    values, indices = torch.topk(prob, k=2, dim=1) # [bsz, 2]
                    indices = indices + 1 # NOTE: shift to the right by 1
                    indices = indices * values.gt(0.5).long() # reset those probability < 0.5
                    two_mask = values.gt(0.5).long().sum(dim=1).eq(2)
                    indices[two_mask] = torch.sort(indices[two_mask], dim=1)[0] # make two dependencies in ascending order
                latest_dep = indices

            # predict inputs
            step_inps = []
            inp_h0 = self.inp_lin(p_word_h)[:, 0].unsqueeze(0) # [1, bsz, dim_h]
            for j in range(self.max_inp): # consider the decoding of i-th step, j-th input
                step_inp = self.inp_decoders[j].generate_sequence(
                    lambda x: self.word_dropout(self.word_embeddings(x)),
                    inp_h0,
                    self.inp_classifiers[j],
                    self.vocab['word_token_to_idx'],
                    self.max_inp_len
                    ) # [bsz, max_inp_len']
                pad_len = self.max_inp_len - step_inp.size(1)
                if pad_len > 0:
                    step_inp = torch.cat([step_inp, torch.zeros(bsz, pad_len).to(device)], dim=1)
                step_inps.append(step_inp)
            step_inps = torch.stack(step_inps, dim=1) # [bsz, 3, max_inp_len]

            history_ph.append(p_word_h)
            programs.append(latest_func)
            dependencies.append(latest_dep)
            inputs.append(step_inps)

            finished = finished | latest_func.eq(end_id).byte()
            if finished.sum().item() == bsz:
                # print('finished at step {}'.format(i))
                break

        programs = torch.stack(programs, dim=1) # [bsz, max_prog]
        dependencies = torch.stack(dependencies, dim=1) # [bsz, max_prog, 2]
        inputs = torch.stack(inputs, dim=1) # [bsz, max_prog, 3, max_inp_len]
        return programs, dependencies, inputs


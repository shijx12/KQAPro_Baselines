import torch
import torch.nn as nn

from utils.BiGRU import GRU, BiGRU
from SRN.knowledge_graph import KnowledgeGraph
from utils.misc import *

class SRN(nn.Module):
    def __init__(self, args, dim_word, dim_hidden, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.num_layers = 2
        self.dim_hidden = 300
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=2, dropout=0.2)
        self.path_encoder = GRU(dim_word, dim_hidden, num_layers=3, dropout = 0.3)
        self.kg = KnowledgeGraph(args, vocab)
        self.num_rollout_steps = args.num_rollout_steps
        self.num_rollouts = args.num_rollouts
        num_words = len(vocab['word2id'])
        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.3)
        self.gamma = self.args.gamma
        self.beta = self.args.beta
        self.eta = self.args.eta
        self.step_encoders = []
        for i in range(self.num_rollout_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m) 
        self.rel_lin = nn.Linear(dim_hidden, 1)
        self.action_classifier = nn.Sequential(
                nn.Linear(2 * dim_hidden, dim_hidden),
                nn.ReLU()
                # nn.Linear(dim_hidden, dim_hidden),
            )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def transit(self, e, q_t, last_h):
        """
        Args:
            e [bsz]
            q_t [bsz, max_q, dim_hidden]
            last_h [bsz, dim_hidden]
        """
        def get_action_space_in_buckets(e):
            device = e.device
            db_action_spaces, db_references = [], []
            entity2bucketid = self.kg.entity2bucketid[e.tolist()]
            key1 = entity2bucketid[:, 0]
            key2 = entity2bucketid[:, 1]
            batch_ref = {}
            for i in range(len(e)):
                key = int(key1[i])
                if not key in batch_ref:
                    batch_ref[key] = []
                batch_ref[key].append(i)
            for key in batch_ref:
                action_space = self.kg.action_space_buckets[key]
                l_batch_refs = batch_ref[key]
                g_bucket_ids = key2[l_batch_refs].tolist()
                r_space_b = action_space[0][0][g_bucket_ids].to(device)
                e_space_b = action_space[0][1][g_bucket_ids].to(device)
                action_mask_b = action_space[1][g_bucket_ids].to(device)
                action_space_b = ((r_space_b, e_space_b), action_mask_b)
                db_action_spaces.append(action_space_b)
                db_references.append(l_batch_refs)
            return db_action_spaces, db_references
            
        db_action_spaces, db_references = get_action_space_in_buckets(e)
        db_outcomes = []
        entropy_list = []
        references = []
        for action_space_b, reference_b in zip(db_action_spaces, db_references):
            q_t_b = q_t[reference_b]
            last_h_b = last_h[reference_b]
            (r_space_b, e_space_b), action_mask_b = action_space_b
            r_space_b_emb = self.kg.relation_embeddings(r_space_b)
            attn_b = torch.softmax(self.rel_lin(r_space_b_emb.unsqueeze(2) * q_t_b.unsqueeze(1)).squeeze(-1), dim = -1)
            q_attn_b = attn_b @ q_t_b
            last_h_b = torch.stack([last_h_b] * q_attn_b.size(1), dim = 1) 
            action_dist_b = r_space_b_emb.unsqueeze(2) @ self.action_classifier(torch.cat((last_h_b, q_attn_b), dim = -1)).unsqueeze(-1)
            action_dist_b = torch.softmax(action_dist_b.squeeze(-1).squeeze(-1), dim = 1) 
            action_dist_b = action_dist_b * action_mask_b + EPSILON
            action_dist_b = action_dist_b / action_dist_b.sum(1, True)
            db_outcomes.append((action_space_b, action_dist_b))
            references.extend(reference_b)
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        return db_outcomes, inv_offset


    def rollout(self, q_word_h, e_s, num_steps = 3):
        """
        Args:
            q_word_h [bsz, max_q, dim_hidden]
            e_s [bsz]
        """
        def sample(db_outcomes, inv_offset):
            next_r_list = []
            next_e_list = []
            log_action_prob_list = []
            entropy_list = []
            for action_space_b, action_dist_b in db_outcomes:
                (r_space_b, e_space_b), action_mask_b = action_space_b
                idx_b = torch.multinomial(action_dist_b, 1, replacement = True)
                next_r_b = torch.gather(r_space_b, 1, idx_b).view(-1)
                next_e_b = torch.gather(e_space_b, 1, idx_b).view(-1)
                action_prob_b = torch.gather(action_dist_b, 1, idx_b).view(-1)
                log_action_prob_b = safe_log(action_prob_b)
                entropy_b = entropy(action_dist_b)
                next_r_list.append(next_r_b)
                next_e_list.append(next_e_b)
                log_action_prob_list.append(log_action_prob_b)
                entropy_list.append(entropy_b)
            next_r = torch.cat(next_r_list, dim = 0)[inv_offset]
            next_e = torch.cat(next_e_list, dim = 0)[inv_offset]
            log_action_prob = torch.cat(log_action_prob_list, dim = 0)[inv_offset]
            policy_entropy = torch.cat(entropy_list, dim = 0)[inv_offset]
            return {
                'next_action': (next_r, next_e),
                'log_action_prob': log_action_prob,
                'policy_entropy': policy_entropy
            }
        
        
        assert num_steps > 0
        bsz = e_s.size(0)
        device = e_s.device
        r_s = torch.LongTensor([self.kg.dummy_start_r] * bsz).to(device)
        last_h = torch.zeros((3, bsz, self.dim_hidden)).float().to(device)
        path_trace = [(r_s, e_s)]
        log_action_probs = []
        action_entropies = []
        hs = []
        qs = []
        for t in range(num_steps):
            last_r, e = path_trace[-1]
            last_r_emb = self.kg.relation_embeddings(last_r).unsqueeze(1) # [bsz, 1, dim_hidden]
            path_emb, last_h = self.path_encoder.forward_one_step(last_r_emb, last_h) # [bsz, 1, dim_hidden] [num_layers, bsz, dim_hidden]
            path_emb = path_emb.squeeze(1)
            q_t = self.step_encoders[t](q_word_h) # [bsz, max_q, dim_hidden]
            qs.append(q_t.sum(1))
            hs.append(path_emb) 
            db_outcomes, inv_offset = self.transit(e, q_t, path_emb)
            output = sample(db_outcomes, inv_offset)
            log_action_probs.append(output['log_action_prob'])
            action_entropies.append(output['policy_entropy'])
            path_trace.append(output['next_action'])
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        relu = nn.ReLU()
        phi = [torch.zeros(bsz, 1).float().to(device)]
        for h, q in zip(hs, qs):
            phi.append(relu(cos(h,q)).unsqueeze(-1))
        phi = torch.cat(phi, dim = 1) # [bsz, num_steps + 1]
        
        pred_e2 = path_trace[-1][1]
        return {
            'pred_e2': pred_e2,
            'log_action_probs':log_action_probs,
            'action_entropies':action_entropies,
            'path_trace': path_trace,
            'phi': phi
        }


    def reward_fun(self, pred_e2, e2):
        """
        Args:
            pred_e2 [bsz]
            e2 [bsz, max_num_answers]
        Return:
            reward [bsz]
        """
        device = e2.device
        return (pred_e2 == e2).float().to(device)
        # return torch.any(pred_e2.unsqueeze(1) == e2, dim=1).float().to(device)
    
    def forward(self, questions, e_s, answers = None, num_rollouts = 3):
        """
        Args:
            questions [bsz, max_q]
            e_s [bsz, 1]
            answers [bsz, max_num_answers]
        """
        num_rollouts = self.args.num_rollouts
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions))
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) 
        e_s = e_s.squeeze()
        if answers == None:
            return self.beam_search(q_word_h, e_s)
        q_word_h, e_s = tile_along_beam(q_word_h, num_rollouts, dim=0), tile_along_beam(e_s, num_rollouts, dim=0)
        output = self.rollout(q_word_h, e_s, num_steps = self.num_rollout_steps)
        verbose = False
        if verbose:
            print('question: {}'.format(' '.join(self.vocab['id2word'].get(word, '<UNK>') for word in questions[0].tolist() if not self.vocab['id2word'][word] == '<PAD>')))
            pt_rs = []
            pt_es = []
            for rs, es in output['path_trace']:
                pt_rs.append(rs[0].item())
                pt_es.append(es[0].item())
            for r, e in zip(pt_rs, pt_es):
                print('{} --> {}'.format(self.vocab['id2relation'][r], self.vocab['id2entity'][e]))

        # Compute policy gradient loss
        answers = tile_along_beam(answers, num_rollouts, dim=0) # [bsz * num_rollouts, max_num_answers]
        pred_e2 = output['pred_e2'] # [bsz * num_rollouts]
        log_action_probs = output['log_action_probs'] 
        action_entropies= output['action_entropies'] # [bsz * num_rollouts]
        phi = output['phi']
        # Compute discounted reward
        final_reward = self.reward_fun(pred_e2, answers) # [bsz * num_rollouts]
        cum_discounted_rewards = [0] * self.num_rollout_steps 
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R
        potential_rewards = [0] * self.num_rollout_steps
        for i in range(self.num_rollout_steps):
            potential_rewards[i] = self.eta * phi[:, (i+1)] - phi[:, i] # [bsz * num_rollouts]
        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i] # [bsz * num_rollouts]
            pg_loss += - (cum_discounted_rewards[i] + potential_rewards[i]) * log_action_prob # [bsz * num_rollouts]
            pt_loss += - (cum_discounted_rewards[i] + potential_rewards[i]) * torch.exp(log_action_prob) # [bsz * num_rollouts]

        # Entropy regularization
        entropy = torch.cat([action_entropy.unsqueeze(1) for action_entropy in action_entropies], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()
        return pg_loss, pt_loss

    
    def beam_search(self, q_word_h, e_s, num_steps = 3, beam_size = 4):
        """
        Args:
            q_word_h [bsz, max_q, dim_hidden]
            e_s [bsz, 1]
        """
        beam_size = self.args.beam_size
        assert num_steps > 0
        bsz = e_s.size(0)
        device = e_s.device

        def top_k_action(log_action_dist, action_space, last_h):
            """
            Args:
                log_action_dist [bsz * k, action_space_size]
                action_space (r_space, e_space), action_mask
                    r_space [bsz * k, action_space_size]
                    e_space [bsz * k, action_space_size]
                last_h [num_layers, bsz * k, dim_hidden]
            """
            assert log_action_dist.size(0) % bsz == 0
            last_k = (int)(log_action_dist.size(0) / bsz)
            r_space, e_space = action_space
            action_space_size = r_space.size(1)
            # [bsz * k, action_space_size] => [bsz, k * action_space_size]
            log_action_dist = log_action_dist.view(bsz, -1)
            # => [bsz, k * action_space_size, dim_hidden]
            dim_hidden = last_h.size()[-1]
            last_h = tile_along_beam(last_h.view(3, bsz, last_k, dim_hidden), action_space_size, dim = 2) # [num_layers, bsz, last_k * action_space_size, dim_hidden]
            k = min(beam_size, log_action_dist.size(1))
            log_action_prob, idx = torch.topk(log_action_dist, k) # idx [bsz, k]
            last_h = torch.gather(last_h, 2, idx.unsqueeze(0).unsqueeze(-1).expand(3, bsz, k, dim_hidden)).view(3, bsz * k, dim_hidden)
            next_r, next_e = torch.gather(action_space[0].view(bsz, -1), 1, idx).view(-1), torch.gather(action_space[1].view(bsz, -1), 1, idx).view(-1) # [num_layers, bsz * k, dim_hidden]
            log_action_prob = torch.gather(log_action_dist, 1, idx).view(-1)
            action_beam_offset = idx // action_space_size # [bsz, k]
            action_batch_offset = (torch.arange(bsz) * last_k).unsqueeze(-1).to(last_h.device) # [bsz, 1]
            action_offset = (action_batch_offset + action_beam_offset).view(-1) # [bsz, k] => [bsz * k]
            return (next_r, next_e), log_action_prob, last_h, action_offset

        
        def pad(db_outcomes, inv_offset):
            r_space_list = []
            e_space_list = []
            action_mask_list = []
            action_dist_list = []
            for action_space_b, action_dist_b in db_outcomes:
                (r_space_b, e_space_b), action_mask_b = action_space_b
                r_space_list.append(r_space_b)
                e_space_list.append(e_space_b)
                action_mask_list.append(action_mask_b)
                action_dist_list.append(action_dist_b)
            r_space = pad_and_cat(r_space_list, padding_value = self.kg.dummy_r)[inv_offset]
            e_space = pad_and_cat(e_space_list, padding_value = self.kg.dummy_e)[inv_offset]
            action_mask = pad_and_cat(action_mask_list, padding_value = 0)[inv_offset]
            action_dist = pad_and_cat(action_dist_list, padding_value = 0)[inv_offset]
            return (r_space, e_space), action_dist

        def adjust_search_trace(search_trace, action_offset):
            for i, (r, e) in enumerate(search_trace):
                new_r = r[action_offset]
                new_e = e[action_offset]
                search_trace[i] = (new_r, new_e)
        r_s = torch.LongTensor([self.kg.dummy_start_r] * bsz).to(device)
        last_h = torch.zeros((3, bsz, self.dim_hidden)).float().to(device)
        log_action_prob = torch.zeros((bsz,)).float().to(device)
        action = (r_s, e_s)
        search_trace = [action]
        for t in range(num_steps):
            last_r, e = action
            last_r_emb = self.kg.relation_embeddings(last_r).unsqueeze(1) # [bsz * k, 1, dim_hidden]
            path_emb, last_h = self.path_encoder.forward_one_step(last_r_emb, last_h)
            path_emb = path_emb.squeeze(1)
            q_t = self.step_encoders[t](q_word_h)
            k = int(e.size(0) / bsz)
            q_t = tile_along_beam(q_t, k, dim = 0) # [bsz * k, max_q, dim_hidden]
            db_outcomes, inv_offset = self.transit(e, q_t, path_emb)
            action_space, action_dist = pad(db_outcomes, inv_offset)
            # action_space, action_dist, _ = self.transit(e, q_t, last_h.squeeze())
            log_action_dist = log_action_prob.view(-1, 1) + safe_log(action_dist)
            action, log_action_prob, last_h, action_offset = top_k_action(log_action_dist, action_space, last_h)
            adjust_search_trace(search_trace, action_offset)
            search_trace.append(action)
        pred_e2s = action[1].view(bsz, -1) # [bsz, beam_size]
        pred_e2_scores = log_action_prob.view(bsz, -1) # [bsz, beam_size]
        return {
            'pred_e2s': pred_e2s,
            'pred_e2_scores': pred_e2_scores,
            'search_traces': search_trace
        }
    
    
 


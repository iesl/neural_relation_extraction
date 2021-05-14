import numpy as np
import json
import torch
import torch.nn as nn

from transformers import AutoModel

from module.neural import TransformerInterEncoder
from module.neural import RNNEncoder
from module.utils import log1mexp

__all__ = [
    "BiaffineNetwork",
    "HierarchyBiaffineNetwork",
    "ConcatNonLinear",
    "HierarchyConcatNonLinear",
    "CenterBox",
    "MinMaxBox",
    "HierarchyMinMaxBox",
    "HierarchyMinMaxBoxHardcoded",
]


def orthonormal_initializer(input_size, output_size):
    """from https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/tf_utils.py#L154"""
    I = np.eye(output_size)
    lr = .1
    eps = .05/(output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI**2 / 2)
            Q2 = Q**2
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return Q.astype(np.float32)


class BiaffineNetwork(nn.Module):
    def __init__(self, config):
        super(BiaffineNetwork, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(open(config["data_path"] + "/relation_map.json").read()))
        self.head_layer0 = torch.nn.Linear(self.D, self.D)
        self.head_layer1 = torch.nn.Linear(self.D, self.dim)
        self.tail_layer0 = torch.nn.Linear(self.D, self.D)
        self.tail_layer1 = torch.nn.Linear(self.D, self.dim)
        self.relu = torch.nn.ReLU()
        mat = orthonormal_initializer(self.dim, self.dim)[:,None,:]
        biaffine_mat = np.concatenate([mat]* (self.num_rel + 1), axis=1)
        self.biaffine_mat = torch.nn.Parameter(torch.tensor(biaffine_mat), requires_grad=True) # (dim, R, dim)
        self.multi_label = config["multi_label"]
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def bi_affine(self, e1_vec, e2_vec):
        # e1_vec: batchsize, text_length, dim
        # e2_vec: batchsize, text_length, dim
        # output: batchsize, text_length, text_length, R
        batchsize, text_length, dim = e1_vec.shape

        # (batchsize * text_length, dim) (dim, R*dim) -> (batchsize * text_length, R*dim)
        lin = torch.matmul(
                torch.reshape(e1_vec, [-1, dim]),
                torch.reshape(self.biaffine_mat, [dim, (self.num_rel + 1) * dim])
        )
        # (batchsize, text_length * R, D) (batchsize, D, text_length) -> (batchsize, text_length * R, text_length)
        bilin = torch.matmul(
            torch.reshape(lin, [batchsize, text_length * (self.num_rel + 1), self.dim]),
            torch.transpose(e2_vec, 1, 2)
        )
        
        output = torch.reshape(bilin, [batchsize, text_length, self.num_rel + 1, text_length])
        output = torch.transpose(output, 2, 3)
        return output
        
    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: (batchsize, text_length, text_length)
        # e1_indicator: not used
        # e2_indicator: not used
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = self.head_layer1(self.relu(self.head_layer0(h)))
        e2_vec = self.tail_layer1(self.relu(self.tail_layer0(h)))


        pairwise_scores = self.bi_affine(e1_vec, e2_vec) # (batchsize, text_length, text_length, R + 1)
        # pairwise_scores = torch.nn.functional.softmax(pairwise_scores, dim=3) 
        # # Commented, was used in original Bran code: https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/models/transformer.py#L468 
        pairwise_scores = pairwise_scores + ep_mask.unsqueeze(3) # batchsize, text_length, text_length, R + 1
        pairwise_scores = torch.logsumexp(pairwise_scores, dim=[1,2]) # batchsize, R + 1
        if self.multi_label == True:
            pairwise_scores = pairwise_scores[:, :-1] # (batchsize, R)

        return pairwise_scores


class HierarchyBiaffineNetwork(BiaffineNetwork):

    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: (batchsize, text_length, text_length)
        # e1_indicator: not used
        # e2_indicator: not used
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = self.head_layer1(self.relu(self.head_layer0(h)))
        e2_vec = self.tail_layer1(self.relu(self.tail_layer0(h)))


        pairwise_scores = self.bi_affine(e1_vec, e2_vec) # (batchsize, text_length, text_length, R)
        # pairwise_scores = torch.nn.functional.softmax(pairwise_scores, dim=3) 
        # # Commented, was used in original Bran code: https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/models/transformer.py#L468 
        pairwise_scores = pairwise_scores + ep_mask.unsqueeze(3) # batchsize, text_length, text_length, R
        pairwise_scores = torch.logsumexp(pairwise_scores, dim=[1,2]) # batchsize, R

        if self.multi_label == False:
            logit_not_na = pairwise_scores[:, -1].unsqueeze(1) # last dimension is score of existing a relation 
            logit_rest = pairwise_scores[:, :-1]
            prob_not_na = self.sigmoid(logit_not_na) # (batchsize, 1)
            prob_rest = prob_not_na * self.softmax(logit_rest) # (batchsize, R)
            prob = torch.cat([prob_rest, 1 - prob_not_na], 1) # (batchsize, R+1)
            logprobs = torch.log(prob)
            return logprobs
        else: # if multilabel is true, no hierarchy.
            pairwise_scores = pairwise_scores[:, :-1] # (batchsize, R)
            return pairwise_scores


class ConcatNonLinear(nn.Module):
    def __init__(self, config):
        super(ConcatNonLinear, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(open(config["data_path"] + "/relation_map.json").read()))
        self.layer1 = torch.nn.Linear(self.D * 2, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.num_rel + 1)
        self.relu = torch.nn.ReLU()
        self.multi_label = config["multi_label"]
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=config["dropout_rate"])
        
    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: not used
        # e1_indicator: (batchsize, text_length)
        # e2_indicator: (batchsize, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = (h * e1_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        e2_vec = (h * e2_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        pairwise_scores = self.layer2(self.dropout(self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1))))) # (batchsize, R+1)
        if self.multi_label == True:
            pairwise_scores = pairwise_scores[:, :-1] # (batchsize, R)
        
        return pairwise_scores


class HierarchyConcatNonLinear(ConcatNonLinear):

    def __init__(self, config):
        super().__init__(config)
        self.layer1_na = torch.nn.Linear(self.D * 2, self.dim)
        self.layer2_na = torch.nn.Linear(self.dim, 1)


    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: not used
        # e1_indicator: (batchsize, text_length)
        # e2_indicator: (batchsize, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = (h * e1_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        e2_vec = (h * e2_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        pairwise_scores = self.layer2(self.dropout(self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1))))) # (batchsize, R + 1)
        
        if self.multi_label == False:
            
            logit_not_na = self.layer2_na(self.dropout(self.relu(self.layer1_na(torch.cat([e1_vec, e2_vec], 1))))) # (batchsize, 1)
            logit_r = pairwise_scores[:, :-1] # (batchsize, R)
            prob_not_na = self.sigmoid(logit_not_na) # (batchsize, 1)
            prob_r = prob_not_na * self.softmax(logit_r) # (batchsize, R)
            
            
            prob = torch.cat([prob_r, 1 - prob_not_na], 1) # (batchsize, R+1)
            log_prob = torch.log(prob)
            return log_prob

        else: # if multilabel is true, no hierarchy.
            pairwise_scores = pairwise_scores[:, :-1] # (batchsize, R)
            return pairwise_scores


class CenterBox(nn.Module):
    def __init__(self, config):
        super(CenterBox, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(open(config["data_path"] + "/relation_map.json").read()))
        self.layer1 = torch.nn.Linear(self.D * 2, self.D * 2)
        self.layer2 = torch.nn.Linear(self.D * 2, self.dim * 2)
        self.relu = torch.nn.ReLU()

        self.rel_center = torch.nn.Parameter(torch.rand(self.num_rel + 1, self.dim) * 0.2 - 0.1, requires_grad=True)
        self.rel_sl = torch.nn.Parameter(torch.zeros(self.num_rel + 1, self.dim), requires_grad=True)
        self.volume_temp = config["volume_temp"]
        self.intersection_temp = config["intersection_temp"]
        self.softplus = torch.nn.Softplus()
        self.softplus_volume = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.multi_label = config["multi_label"]
        self.hsm = config["hsm"]
        self.softmax = torch.nn.Softmax(dim=1)
    

    def transition_matrix(self):
        min_, max_ = self.param2minmax(self.rel_center, self.rel_sl) # (R+1, D)
        row_z = min_[:, None, :].repeat(1, self.num_rel + 1, 1) # (R+1, R+1, D)
        row_Z = max_[:, None, :].repeat(1, self.num_rel + 1, 1)
        col_z = min_[None, :, :].repeat(self.num_rel + 1, 1, 1)
        col_Z = max_[None, :, :].repeat(self.num_rel + 1, 1, 1)

        meet_z, meet_Z = self.gumbel_intersection(row_z, row_Z, col_z, col_Z) # (R+1, R+1, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R+1, R+1, )
        log_row_volume = self.log_volume(row_z, row_Z) # (R+1, R+1, )
        log_prob = log_overlap_volume - log_row_volume # (R+1, R+1, )
        return log_prob, log_row_volume[:, 0] # log P(column | row) = log_prob[row, column]

    def prob_not_na_given_r(self):
        min_, max_ = self.param2minmax(self.rel_center, self.rel_sl) # (R+1, D)
        r_z = min_[:-1] # (R, D)
        r_Z = max_[:-1] # (R, D)
        not_na_z = min_[-1:].repeat(self.num_rel, 1) # (R, D)
        not_na_Z = max_[-1:].repeat(self.num_rel, 1) # (R, D)

        meet_z, meet_Z = self.gumbel_intersection(not_na_z, not_na_Z, r_z, r_Z) # (R, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R, )
        log_row_volume = self.log_volume(r_z, r_Z) # (R, )
        log_prob = log_overlap_volume - log_row_volume # (R, )
        return log_prob # log P(not_na | r) = log_prob[r]


    def param2minmax(self, center, length):
        length_ = self.softplus(length)
        z = center - length_
        Z = center + length_
        return z, Z


    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.softplus_volume(Z - z - self.softplus_const)),
            dim=-1,
        )
        return log_vol


    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max


    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: not used
        # e1_indicator: (batchsize, text_length)
        # e2_indicator: (batchsize, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = (h * e1_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        e2_vec = (h * e2_indicator[:,:,None]).max(1)[0] # (batchsize, D)


        input_boxes = self.layer2(self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1)))) # (batchsize, 2*D)
        input_z, input_Z = self.param2minmax(input_boxes[:, :self.dim], input_boxes[:, self.dim:]) # (batchsize, D), (batchsize, D)
        input_z, input_Z = input_z[:, None, :].repeat(1, self.num_rel + 1, 1), input_Z[:, None, :].repeat(1, self.num_rel + 1, 1) # (batchsize, R + 1, D), (batchsize, R + 1, D)
        rel_z, rel_Z = self.param2minmax(self.rel_center, self.rel_sl) # (R + 1, D), (R + 1, D)
        rel_z, rel_Z = rel_z[None, :, :].repeat(batchsize, 1, 1), rel_Z[None, :, :].repeat(batchsize, 1, 1) # (batchsize, R + 1, D), (batchsize, R + 1, D)

        meet_z, meet_Z = self.gumbel_intersection(input_z, input_Z, rel_z, rel_Z) # (batchsize, R + 1, D), (batchsize, R + 1, D)

        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (batchsize, R + 1)
        log_rhs_volume = self.log_volume(input_z, input_Z) # (batchsize, R + 1)

        if self.multi_label == True:
            log_prob = log_overlap_volume[:, :-1] - log_rhs_volume[:, :-1] # (batchsize, R)
        else:
            log_prob = log_overlap_volume # (batchsize, R+1)
        # elif self.multi_label == False:
        #     logprob_not_na = (log_overlap_volume[:, -1] - log_rhs_volume[:, -1]).unsqueeze(1) # last dimension is score of existing a relation 
        #     logprob_rest = log_overlap_volume[:, :-1]

        #     logprob_na = log1mexp(logprob_not_na) # (batchsize, 1)
        #     logprob_rest = logprob_not_na + torch.log(self.softmax(logprob_rest)) # (batchsize, R-1)
        #     log_prob = torch.cat([logprob_rest, logprob_na], 1) # (batchsize, R)
            
        return log_prob


class MinMaxBox(nn.Module):
    def __init__(self, config):
        super(MinMaxBox, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(open(config["data_path"] + "/relation_map.json").read()))

        self.layer1 = torch.nn.Linear(self.D * 2, self.D * 2)
        self.layer2 = torch.nn.Linear(self.D * 2, self.dim * 2)
        self.relu = torch.nn.ReLU()

        self.rel_min = torch.nn.Parameter(torch.rand(self.num_rel+1, self.dim) * 0.1 - 0.1, requires_grad=True)
        self.rel_max = torch.nn.Parameter(torch.rand(self.num_rel+1, self.dim) * 0.1, requires_grad=True)
        self.volume_temp = config["volume_temp"]
        self.intersection_temp = config["intersection_temp"]
        self.softplus_volume = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.multi_label = config["multi_label"]
        self.softmax = torch.nn.Softmax(dim=1)
        
    def transition_matrix(self):
        min_, max_ = self.rel_min, self.rel_max # (R+1, D)
        row_z = min_[:, None, :].repeat(1, self.num_rel + 1, 1) # (R+1, R+1, D)
        row_Z = max_[:, None, :].repeat(1, self.num_rel + 1, 1)
        col_z = min_[None, :, :].repeat(self.num_rel + 1, 1, 1)
        col_Z = max_[None, :, :].repeat(self.num_rel + 1, 1, 1)

        meet_z, meet_Z = self.gumbel_intersection(row_z, row_Z, col_z, col_Z) # (R+1, R+1, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R+1, R+1, )
        log_row_volume = self.log_volume(row_z, row_Z) # (R+1, R+1, )
        log_prob = log_overlap_volume - log_row_volume # (R+1, R+1, )
        return log_prob, log_row_volume[:, 0] # log P(column | row) = log_prob[row, column]

    
    def prob_not_na_given_r(self):
        min_, max_ = self.rel_min, self.rel_max # (R+1, D)
        r_z = min_[:-1] # (R, D)
        r_Z = max_[:-1] # (R, D)
        not_na_z = min_[-1:].repeat(self.num_rel, 1) # (R, D)
        not_na_Z = max_[-1:].repeat(self.num_rel, 1) # (R, D)

        meet_z, meet_Z = self.gumbel_intersection(not_na_z, not_na_Z, r_z, r_Z) # (R, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R, )
        log_row_volume = self.log_volume(r_z, r_Z) # (R, )
        log_prob = log_overlap_volume - log_row_volume # (R, )
        return log_prob # log P(not_na | r) = log_prob[r]
    

    def param2minmax(self, min_, max_):
        z = torch.min(min_, max_)
        Z = torch.max(min_, max_)
        return z, Z


    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.softplus_volume(Z - z - self.softplus_const)),
            dim=-1,
        )
        return log_vol

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: not used
        # e1_indicator: (batchsize, text_length)
        # e2_indicator: (batchsize, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = (h * e1_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        e2_vec = (h * e2_indicator[:,:,None]).max(1)[0] # (batchsize, D)


        input_boxes = self.layer2(self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1)))) # (batchsize, 2*D)
        input_z, input_Z = self.param2minmax(input_boxes[:, :self.dim], input_boxes[:, self.dim:]) # (batchsize, D), (batchsize, D)
        input_z, input_Z = input_z[:, None, :].repeat(1, self.num_rel + 1, 1), input_Z[:, None, :].repeat(1, self.num_rel + 1, 1) # (batchsize, R+1, D), (batchsize, R+1, D)
        rel_z, rel_Z = self.param2minmax(self.rel_min, self.rel_max) # (R+1, D), (R+1, D)
        rel_z, rel_Z = rel_z[None, :, :].repeat(batchsize, 1, 1), rel_Z[None, :, :].repeat(batchsize, 1, 1) # (batchsize, R+1, D), (batchsize, R+1, D)

        meet_z, meet_Z = self.gumbel_intersection(input_z, input_Z, rel_z, rel_Z) # (batchsize, R+1, D), (batchsize, R+1, D)

        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (batchsize, R+1)
        log_rhs_volume = self.log_volume(input_z, input_Z) # (batchsize, R+1)

        if self.multi_label == True:
            log_prob = log_overlap_volume[:, :-1] - log_rhs_volume[:, :-1] # (batchsize, R)
        else:
            log_prob = log_overlap_volume # (batchsize, R + 1)
            
        return log_prob


class HierarchyMinMaxBox(MinMaxBox):

    def __init__(self, config):
        super().__init__(config)

        self.rel_min = torch.nn.Parameter(torch.rand(self.num_rel, self.dim) * 0.1 - 0.1, requires_grad=True)
        self.rel_max = torch.nn.Parameter(torch.rand(self.num_rel, self.dim) * 0.1, requires_grad=True)
        self.na_min = torch.nn.Parameter(torch.rand(1, self.dim)*0.5 - 1.0, requires_grad=True) # [-1, -0.5]
        self.na_max = torch.nn.Parameter(torch.rand(1, self.dim)*0.5 + 0.5, requires_grad=True) # [0.5, 1.0]


    def relation_box(self):
        
        z_na, Z_na = self.param2minmax(self.na_min, self.na_max) # (1, D)
        z_r, Z_r = self.param2minmax(self.rel_min, self.rel_max)
        
        z = torch.cat([z_r, z_na], 0) # (R+1, D)
        Z = torch.cat([Z_r, Z_na], 0)

        return z, Z

    def transition_matrix(self):
        min_, max_ = self.relation_box() # (R+1, D)
        row_z = min_[:, None, :].repeat(1, self.num_rel + 1, 1) # (R+1, R+1, D)
        row_Z = max_[:, None, :].repeat(1, self.num_rel + 1, 1)
        col_z = min_[None, :, :].repeat(self.num_rel + 1, 1, 1)
        col_Z = max_[None, :, :].repeat(self.num_rel + 1, 1, 1)

        meet_z, meet_Z = self.gumbel_intersection(row_z, row_Z, col_z, col_Z) # (R+1, R+1, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R+1, R+1, )
        log_row_volume = self.log_volume(row_z, row_Z) # (R+1, R+1, )
        log_prob = log_overlap_volume - log_row_volume # (R+1, R+1, )
        return log_prob, log_row_volume[:, 0] # log P(column | row) = log_prob[row, column]


    def prob_not_na_given_r(self):
        min_, max_ = self.relation_box() # (R+1, D)
        r_z = min_[:-1] # (R, D)
        r_Z = max_[:-1] # (R, D)
        not_na_z = min_[-1:].repeat(self.num_rel, 1) # (R, D)
        not_na_Z = max_[-1:].repeat(self.num_rel, 1) # (R, D)

        meet_z, meet_Z = self.gumbel_intersection(not_na_z, not_na_Z, r_z, r_Z) # (R, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (R, )
        log_row_volume = self.log_volume(r_z, r_Z) # (R, )
        log_prob = log_overlap_volume - log_row_volume # (R, )
        return log_prob # log P(not_na | r) = log_prob[r]


    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask, e1_indicator, e2_indicator):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: not used
        # e1_indicator: (batchsize, text_length)
        # e2_indicator: (batchsize, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = (h * e1_indicator[:,:,None]).max(1)[0] # (batchsize, D)
        e2_vec = (h * e2_indicator[:,:,None]).max(1)[0] # (batchsize, D)


        input_boxes = self.layer2(self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1)))) # (batchsize, 2*D)
        input_z, input_Z = self.param2minmax(input_boxes[:, :self.dim], input_boxes[:, self.dim:]) # (batchsize, D), (batchsize, D)
        input_z, input_Z = input_z[:, None, :].repeat(1, self.num_rel+1, 1), input_Z[:, None, :].repeat(1, self.num_rel+1, 1) # (batchsize, R+1, D), (batchsize, R+1, D)
        rel_z, rel_Z = self.relation_box() # (R+1, D), (R+1, D)
        rel_z, rel_Z = rel_z[None, :, :].repeat(batchsize, 1, 1), rel_Z[None, :, :].repeat(batchsize, 1, 1) # (batchsize, R+1, D), (batchsize, R+1, D)

        meet_z, meet_Z = self.gumbel_intersection(input_z, input_Z, rel_z, rel_Z) # (batchsize, R+1, D), (batchsize, R+1, D)

        log_overlap_volume = self.log_volume(meet_z, meet_Z) # (batchsize, R+1)
        log_rhs_volume = self.log_volume(input_z, input_Z) # (batchsize, R+1)

        if self.multi_label == True:
            log_prob = log_overlap_volume[:, :-1] - log_rhs_volume[:, :-1]  # (batchsize, R)
        else:
            log_cond_prob_not_na = (log_overlap_volume[:, -1] - log_rhs_volume[:, -1]).unsqueeze(1) # last dimension is log prob of existing some relation
            log_cond_prob_na = log1mexp(log_cond_prob_not_na) # (batchsize, 1)
            log_prob_r = log_cond_prob_not_na + torch.log(self.softmax(log_overlap_volume[:, :-1])) # (batchsize, R)
            log_prob = torch.cat([log_prob_r, log_cond_prob_na], 1) # (batchsize, R + 1)
        return log_prob


class HierarchyMinMaxBoxHardcoded(HierarchyMinMaxBox):

    def __init__(self, config):
        super().__init__(config)

        self.rel_min = torch.nn.Parameter(torch.rand(self.num_rel, self.dim) * 0.5 - 0.5, requires_grad=True)
        self.rel_max = torch.nn.Parameter(torch.rand(self.num_rel, self.dim) * 0.5, requires_grad=True)
        self.sigmoid = torch.nn.Sigmoid()


    def relation_box(self):
        
        z_na, Z_na = self.param2minmax(self.na_min, self.na_max) # (1, D)
        z_r, Z_r = self.param2minmax(self.rel_min, self.rel_max)
        z_r, Z_r = self.sigmoid(z_r), self.sigmoid(Z_r) # (R, D)
        
        z_r, Z_r = z_na + z_r * (Z_na - z_na), z_na + Z_r * (Z_na - z_na) # (R, D)
        
        z = torch.cat([z_r, z_na], 0) # (R+1, D)
        Z = torch.cat([Z_r, Z_na], 0)

        return z, Z

    

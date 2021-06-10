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
    "ConcatNonLinear",
    "ConcatNonLinearNormalized",
    "MinMaxBox",
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
            Q -= lr*Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0,
                                                       keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
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
        self.num_rel = len(json.loads(
            open(config["data_path"] + "/relation_map.json").read()))
        self.head_layer0 = torch.nn.Linear(self.D, self.D)
        self.head_layer1 = torch.nn.Linear(self.D, self.dim)
        self.tail_layer0 = torch.nn.Linear(self.D, self.D)
        self.tail_layer1 = torch.nn.Linear(self.D, self.dim)
        self.relu = torch.nn.ReLU()
        mat = orthonormal_initializer(self.dim, self.dim)[:, None, :]
        biaffine_mat = np.concatenate([mat] * (self.num_rel + 1), axis=1)
        self.biaffine_mat = torch.nn.Parameter(torch.tensor(
            biaffine_mat), requires_grad=True)  # (dim, R, dim)
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
            torch.reshape(lin, [batchsize, text_length *
                          (self.num_rel + 1), self.dim]),
            torch.transpose(e2_vec, 1, 2)
        )

        output = torch.reshape(
            bilin, [batchsize, text_length, self.num_rel + 1, text_length])
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
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(
        ), attention_mask=attention_mask.long())[0]  # (batchsize, text_length, D)

        e1_vec = self.head_layer1(self.relu(self.head_layer0(h)))
        e2_vec = self.tail_layer1(self.relu(self.tail_layer0(h)))

        # (batchsize, text_length, text_length, R + 1)
        pairwise_scores = self.bi_affine(e1_vec, e2_vec)
        # pairwise_scores = torch.nn.functional.softmax(pairwise_scores, dim=3)
        # # Commented, was used in original Bran code: https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/models/transformer.py#L468
        # batchsize, text_length, text_length, R + 1
        pairwise_scores = pairwise_scores + ep_mask.unsqueeze(3)
        pairwise_scores = torch.logsumexp(
            pairwise_scores, dim=[1, 2])  # batchsize, R + 1
        if self.multi_label == True:
            pairwise_scores = pairwise_scores[:, :-1]  # (batchsize, R)

        return pairwise_scores, None


class ConcatNonLinear(nn.Module):
    def __init__(self, config):
        super(ConcatNonLinear, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(
            open(config["data_path"] + "/relation_map.json").read()))
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
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(
        ), attention_mask=attention_mask.long())[0]  # (batchsize, text_length, D)

        # e1_vec = (h * e1_indicator[:, :, None]).max(1)[0]  # (batchsize, D)
        # e2_vec = (h * e2_indicator[:, :, None]).max(1)[0]  # (batchsize, D)

        e1_indicator_mask = 1000 * \
            torch.ones_like(e1_indicator) * (1 - e1_indicator)
        e2_indicator_mask = 1000 * \
            torch.ones_like(e2_indicator) * (1 - e2_indicator)
        e1_vec = (h * e1_indicator[:, :, None] -
                  e1_indicator_mask[:, :, None]).max(1)[0]  # (batchsize, D)
        e2_vec = (h * e2_indicator[:, :, None] -
                  e2_indicator_mask[:, :, None]).max(1)[0]  # (batchsize, D)

        e1e2_vec = self.dropout(
            self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1))))
        pairwise_scores = self.layer2(e1e2_vec)  # (batchsize, R+1)
        if self.multi_label == True:
            pairwise_scores = pairwise_scores[:, :-1]  # (batchsize, R)

        return pairwise_scores, e1e2_vec


class ConcatNonLinearNormalized(nn.Module):
    def __init__(self, config):
        super(ConcatNonLinearNormalized, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(
            open(config["data_path"] + "/relation_map.json").read()))
        self.layer1 = torch.nn.Linear(self.D * 2, self.dim)
        relation_mat = (torch.rand(self.dim, self.num_rel + 1)
                        * 2 - 1) / (self.dim ** 0.5)
        relation_mat = relation_mat / \
            torch.norm(relation_mat, dim=0, p=2, keepdim=True)  # D, R + 1
        self.relation_mat = torch.nn.Parameter(relation_mat)
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
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(
        ), attention_mask=attention_mask.long())[0]  # (batchsize, text_length, D)

        # e1_vec = (h * e1_indicator[:, :, None]).max(1)[0]  # (batchsize, D)
        # e2_vec = (h * e2_indicator[:, :, None]).max(1)[0]  # (batchsize, D)

        e1_indicator_mask = 1000 * \
            torch.ones_like(e1_indicator) * (1 - e1_indicator)
        e2_indicator_mask = 1000 * \
            torch.ones_like(e2_indicator) * (1 - e2_indicator)
        e1_vec = (h * e1_indicator[:, :, None] -
                  e1_indicator_mask[:, :, None]).max(1)[0]  # (batchsize, D)
        e2_vec = (h * e2_indicator[:, :, None] -
                  e2_indicator_mask[:, :, None]).max(1)[0]  # (batchsize, D)

        e1e2_vec = self.dropout(
            self.relu(self.layer1(torch.cat([e1_vec, e2_vec], 1))))  # batchsize, D
        # normalization
        e1e2_vec_normalized = e1e2_vec / \
            torch.norm(e1e2_vec, dim=1, p=2, keepdim=True)  # batchsize, D
        relation_mat_normalized = self.relation_mat / \
            torch.norm(self.relation_mat, dim=0, p=2, keepdim=True)  # D, R + 1
        # pairwise_scores = self.layer2(e1e2_vec)  # (batchsize, R+1)
        pairwise_scores = torch.matmul(
            e1e2_vec_normalized, relation_mat_normalized)  # batchsize, R + 1
        if self.multi_label == True:
            pairwise_scores = pairwise_scores[:, :-1]  # (batchsize, R)

        return pairwise_scores, e1e2_vec_normalized


class MinMaxBox(nn.Module):
    def __init__(self, config):
        super(MinMaxBox, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = self.encoder.config.hidden_size
        self.dim = config["dim"]
        self.num_rel = len(json.loads(
            open(config["data_path"] + "/relation_map.json").read()))

        self.layer1 = torch.nn.Linear(self.D * 2, self.D * 2)
        self.layer2 = torch.nn.Linear(self.D * 2, self.dim * 2)
        self.relu = torch.nn.ReLU()

        self.rel_min = torch.nn.Parameter(torch.rand(
            self.num_rel+1, self.dim) * 0.1 - 0.1, requires_grad=True)
        self.rel_max = torch.nn.Parameter(torch.rand(
            self.num_rel+1, self.dim) * 0.1, requires_grad=True)
        self.volume_temp = config["volume_temp"]
        self.intersection_temp = config["intersection_temp"]
        self.softplus_volume = torch.nn.Softplus(beta=1 / self.volume_temp)
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.multi_label = config["multi_label"]
        self.softmax = torch.nn.Softmax(dim=1)

    def transition_matrix(self):
        min_, max_ = self.rel_min, self.rel_max  # (R+1, D)
        row_z = min_[:, None, :].repeat(
            1, self.num_rel + 1, 1)  # (R+1, R+1, D)
        row_Z = max_[:, None, :].repeat(1, self.num_rel + 1, 1)
        col_z = min_[None, :, :].repeat(self.num_rel + 1, 1, 1)
        col_Z = max_[None, :, :].repeat(self.num_rel + 1, 1, 1)

        meet_z, meet_Z = self.gumbel_intersection(
            row_z, row_Z, col_z, col_Z)  # (R+1, R+1, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z)  # (R+1, R+1, )
        log_row_volume = self.log_volume(row_z, row_Z)  # (R+1, R+1, )
        log_prob = log_overlap_volume - log_row_volume  # (R+1, R+1, )
        # log P(column | row) = log_prob[row, column]
        return log_prob, log_row_volume[:, 0]

    def prob_not_na_given_r(self):
        min_, max_ = self.rel_min, self.rel_max  # (R+1, D)
        r_z = min_[:-1]  # (R, D)
        r_Z = max_[:-1]  # (R, D)
        not_na_z = min_[-1:].repeat(self.num_rel, 1)  # (R, D)
        not_na_Z = max_[-1:].repeat(self.num_rel, 1)  # (R, D)

        meet_z, meet_Z = self.gumbel_intersection(
            not_na_z, not_na_Z, r_z, r_Z)  # (R, )
        log_overlap_volume = self.log_volume(meet_z, meet_Z)  # (R, )
        log_row_volume = self.log_volume(r_z, r_Z)  # (R, )
        log_prob = log_overlap_volume - log_row_volume  # (R, )
        return log_prob  # log P(not_na | r) = log_prob[r]

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
                [-e1_max / self.intersection_temp, -
                    e2_max / self.intersection_temp]
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
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(
        ), attention_mask=attention_mask.long())[0]  # (batchsize, text_length, D)

        e1_vec = (h * e1_indicator[:, :, None]).max(1)[0]  # (batchsize, D)
        e2_vec = (h * e2_indicator[:, :, None]).max(1)[0]  # (batchsize, D)

        input_boxes = self.layer2(self.relu(self.layer1(
            torch.cat([e1_vec, e2_vec], 1))))  # (batchsize, 2*D)
        input_z, input_Z = self.param2minmax(
            input_boxes[:, :self.dim], input_boxes[:, self.dim:])  # (batchsize, D), (batchsize, D)
        input_z, input_Z = input_z[:, None, :].repeat(1, self.num_rel + 1, 1), input_Z[:, None, :].repeat(
            1, self.num_rel + 1, 1)  # (batchsize, R+1, D), (batchsize, R+1, D)
        rel_z, rel_Z = self.param2minmax(
            self.rel_min, self.rel_max)  # (R+1, D), (R+1, D)
        rel_z, rel_Z = rel_z[None, :, :].repeat(batchsize, 1, 1), rel_Z[None, :, :].repeat(
            batchsize, 1, 1)  # (batchsize, R+1, D), (batchsize, R+1, D)

        # (batchsize, R+1, D), (batchsize, R+1, D)
        meet_z, meet_Z = self.gumbel_intersection(
            input_z, input_Z, rel_z, rel_Z)

        log_overlap_volume = self.log_volume(
            meet_z, meet_Z)  # (batchsize, R+1)
        log_rhs_volume = self.log_volume(input_z, input_Z)  # (batchsize, R+1)

        if self.multi_label == True:
            log_prob = log_overlap_volume[:, :-1] - \
                log_rhs_volume[:, :-1]  # (batchsize, R)
        else:
            log_prob = log_overlap_volume  # (batchsize, R + 1)

        return log_prob, None

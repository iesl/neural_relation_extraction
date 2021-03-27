import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from module.neural import TransformerInterEncoder
from module.neural import RNNEncoder

__all__ = [
    "BRAN",
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


class BRAN(nn.Module):
    def __init__(self, config):
        super(BRAN, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config["encoder_type"])
        self.D = 768
        self.num_rel = 14
        self.head_layer0 = torch.nn.Linear(self.D, self.D)
        self.head_layer1 = torch.nn.Linear(self.D, self.D)
        self.tail_layer0 = torch.nn.Linear(self.D, self.D)
        self.tail_layer1 = torch.nn.Linear(self.D, self.D)
        self.relu = torch.nn.ReLU()
        mat = orthonormal_initializer(self.D, self.D)[:,None,:]
        biaffine_mat = np.concatenate([mat]*self.num_rel, axis=1)
        self.biaffine_mat = torch.nn.Parameter(torch.tensor(biaffine_mat), requires_grad=True) # (D, R, D)

    def bi_affine(self, e1_vec, e2_vec):
        # e1_vec: batchsize, text_length, D
        # e2_vec: batchsize, text_length, D
        # output: batchsize, text_length, text_length, R
        batchsize, text_length, dim = e1_vec.shape

        # (batchsize * text_length, D) (D, R*D) -> (batchsize * text_length, R*D)
        lin = torch.matmul(
                torch.reshape(e1_vec, [-1, dim]),
                torch.reshape(self.biaffine_mat, [dim, self.num_rel * dim])
        )
        # (batchsize, text_length * R, D) (batchsize, D, text_length) -> (batchsize, text_length * R, text_length)
        bilin = torch.matmul(
            torch.reshape(lin, [batchsize, text_length * self.num_rel, self.D]),
            torch.transpose(e2_vec, 1, 2)
        )
        
        output = torch.reshape(bilin, [batchsize, text_length, self.num_rel, text_length])
        output = torch.transpose(output, 2, 3)
        return output
        
    def forward(self, input_ids, token_type_ids, attention_mask, ep_mask):
        # input_ids: (batchsize, text_length)
        # subj_indicators: (batchsize, text_length)
        # obj_indicators: (batchsize, text_length)
        # ep_mask: (batchsize, text_length, text_length)
        batchsize, text_length = input_ids.shape
        h = self.encoder(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long())[0] # (batchsize, text_length, D)
        
        e1_vec = self.head_layer0(self.relu(self.head_layer0(h)))
        e2_vec = self.tail_layer1(self.relu(self.tail_layer1(h)))


        pairwise_scores = self.bi_affine(e1_vec, e2_vec) # (batchsize, text_length, text_length, R)
        # pairwise_scores = torch.nn.functional.softmax(pairwise_scores, dim=3) 
        # # Commented, was used in original Bran code: https://github.com/patverga/bran/blob/32378da8ac339393d9faa2ff2d50ccb3b379e9a2/src/models/transformer.py#L468 
        pairwise_scores = pairwise_scores + ep_mask.unsqueeze(3) # batchsize, text_length, text_length, R
        pairwise_scores = torch.logsumexp(pairwise_scores, dim=[1,2]) # batchsize, R
        return pairwise_scores

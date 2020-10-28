# -*- coding: utf-8 -*-
# !@time: 2020/10/27 下午3:35
# !@author: superMC @email: 18758266469@163.com
# !@fileName: transformer.py
from abc import ABC
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model.utils import get_pad_mask, get_subsequent_mask
from model.layers import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module, ABC):
    def __init__(self, d_hidden, n_position=200):
        super().__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hidden))

    def _get_sinusoid_encoding_table(self, n_position, d_hidden):
        """ Sinusoid position encoding table """

        # TODO: make it with torch instead of numpy

        def get_psoition_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hidden) for hid_j in range(d_hidden)]

        sinusoid_table = np.array([get_psoition_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module, ABC):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_emb = self.src_word_emb(src_seq)
        enc_output = self.dropout(self.position_enc(enc_emb))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module, ABC):
    """ A decoder model with self attention mechanism. """

    def __init__(
            self, n_target_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.target_word_emb = nn.Embedding(n_target_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, target_seq, target_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_emb = self.target_word_emb(target_seq)
        dec_output = self.dropout(self.position_enc(dec_emb))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=target_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module, ABC):
    def __init__(self, n_src_vocab, n_target_vocab, src_pad_idx, target_pad_idx, d_word_vec=512, d_model=512,
                 d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, target_emb_project_weight_sharing=True,
                 src_emb_project_weight_sharing=True):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_target_vocab=n_target_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=target_pad_idx, dropout=dropout)

        self.target_word_project = nn.Linear(d_model, n_target_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if target_emb_project_weight_sharing:
            self.target_word_project.weight = self.decoder.target_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        if src_emb_project_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.target_word_emb.weight

    def forward(self, src_seq, target_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        target_mask = get_pad_mask(target_seq, self.target_pad_idx) & get_subsequent_mask(target_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(target_seq, target_mask, enc_output, src_mask)

        seq_logit = self.target_word_project(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))


if __name__ == '__main__':
    transformer = Transformer(
        9521,
        9521,
        src_pad_idx=1,
        target_pad_idx=1,
        target_emb_project_weight_sharing=True,
        src_emb_project_weight_sharing=True,
        d_k=64,
        d_v=64,
        d_model=512,
        d_word_vec=512,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        dropout=0.1)
    n = 9521  # 类别数
    input_tensor = torch.randint(0, n, size=(2, 10))  # 生成数组元素0~5的二维数组（15*15）
    output_tensor = torch.randint(0, n, size=(2, 10))
    pred = transformer(input_tensor, output_tensor)
    print(pred)

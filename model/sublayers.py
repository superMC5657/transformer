# -*- coding: utf-8 -*-
# !@time: 2020/10/27 下午2:25
# !@author: superMC @email: 18758266469@163.com
# !@fileName: sublayers.py
from abc import ABC

from torch import nn

from model.modules import ScaledDotProductAttention


class PositionwiseFeedForward(nn.Module, ABC):
    def __init__(self, d_in, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        residual = x
        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module, ABC):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_v = d_v
        self.d_k = d_k
        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # 不能直接reshape到(sz_b,n_head,len_q,d_k) 因为q本身sz_b,len_q,d_model d_model-> n_head * d_q
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn

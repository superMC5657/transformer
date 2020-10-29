# -*- coding: utf-8 -*-
# !@time: 2020/10/27 下午3:25
# !@author: superMC @email: 18758266469@163.com
# !@fileName: modules.py
from abc import ABC

from torch import nn, matmul


class ScaledDotProductAttention(nn.Module, ABC):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout, inplace=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(self.softmax(attn))
        # dropout 前softmax的话 inplace必须为false
        output = matmul(attn, v)

        return output, attn

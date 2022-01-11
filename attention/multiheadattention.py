import torch.nn as nn
import torch
import torch.functional as F
import math
from .single import Attention

class Multiheadattention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        print(d_model)
        print(h)
        assert d_model % h == 0

        self.dk = d_model//h
        self.h = h

        self.linearlayers = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(3)])
        self.outputlayer = nn.Linear(d_model, d_model)
        self.Attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, k, mask=None):

        batchsize = k.size(0)
        query, key, value = [l(x).view(batchsize, -1, self.h, self.dk).transpose(1, 2)
                            for l, x in zip(self.linearlayers, (k, k, k))]

        x, attention = self.Attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batchsize, -1, self.h*self.dk)

        return self.outputlayer(x)
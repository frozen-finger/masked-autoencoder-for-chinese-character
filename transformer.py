from attention.multiheadattention import Multiheadattention
from attention.single import Attention
from utils.FeedForward import fc
from utils.Norm import Normal
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, dim, heads, h_dim, h_forward, depth, dropout=0.1):
        super(Transformer, self).__init__()
        self.layerlist = nn.ModuleList([])
        for _ in range(depth):
            self.layerlist.append(nn.ModuleList([Normal(dim, Multiheadattention(heads, dim, dropout)), Normal(dim, fc(dim, h_forward, dropout))]))

    def forward(self, x):
        for att, fc in self.layerlist:
            x = att(x) + x
            x = fc(x) + x
        return x
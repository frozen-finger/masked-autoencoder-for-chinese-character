import torch
import torch.nn as nn
from transformer import Transformer


class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, heads, h_dim, h_feedforward, depth, dropout=0.):
        super(Encoder, self).__init__()
        self.s = patch_size
        self.tranformer = Transformer(dim, heads, h_dim, h_feedforward, depth, dropout)

    def forward(self, x):
        x = self.tranformer(x)
        return x


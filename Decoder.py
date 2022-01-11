import torch
import torch.nn as nn
from transformer import Transformer

class Decoder(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, h_feedforward, patch_size, depth, num_heads, drop_rate=0., ):
        super(Decoder, self).__init__()
        self.dim = dim
        self.tranformer = Transformer(dim, num_heads, hidden_dim, h_feedforward, depth, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = self.tranformer(x)
        x = self.linear(x)
        return x

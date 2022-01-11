import torch
import torch.nn as nn

class Normal(nn.Module):
    def __init__(self, dim, fn):
        super(Normal, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x_ = self.norm(x)
        return self.fn(x_, **kwargs)

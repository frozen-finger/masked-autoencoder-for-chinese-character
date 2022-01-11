import torch
import torch.nn as nn

class Positionembedding(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(Positionembedding, self).__init__()

    def forward(self, x):
        pos_emb = nn.Parameter(torch.arange(1, x.size(1)+1).float())
        pos_emb = pos_emb.unsqueeze(1)
        pos_emb = pos_emb.expand(x.size(1), self.dim)
        pos_emb = pos_emb.unsqueeze(0)
        return pos_emb

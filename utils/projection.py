import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, patch_size, dim):
        super(Projection, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.projector = nn.Linear(3*patch_size*patch_size, dim)
    
    def forward(self, x):
        x = self.projector(x)
        return x
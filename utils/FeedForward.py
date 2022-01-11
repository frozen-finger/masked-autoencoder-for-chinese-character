import torch
import torch.nn as nn

class fc(nn.Module):
    def __init__(self, dim, hidden, dropout=0.):
        super(fc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
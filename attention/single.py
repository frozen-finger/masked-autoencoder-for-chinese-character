import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def forward(self, query, Key, values, mask=None, dropout=None):
        scores = torch.matmul(query, Key.transpose(-2, -1))/math.sqrt(values.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, values), p_attn

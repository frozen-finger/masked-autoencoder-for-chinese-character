import torch
import torch.nn as nn
from vit_pytorch import ViT

class vitmodel(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super(vitmodel, self).__init__()
        self.vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
    def forward(self, x):
        return self.vit(x)
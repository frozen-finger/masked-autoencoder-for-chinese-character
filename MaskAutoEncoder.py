import torch
import torch.nn as nn
from Embedding.PostionEmbedding import Positionembedding
from Decoder import Decoder
from Encoder import Encoder
from utils.projection import Projection
from utils.vitmask import Mask
from utils.PatchGeneration import PatchEmb


class MaskedAutoEncoder(nn.Module):
    def __init__(self, dim, h_dim, h_feedforward, head_encoder, head_decoder, depth_encoder, depth_decoder, image_size, patch_size):
        super(MaskedAutoEncoder, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.projector = Projection(patch_size, dim)
        self.num_patch = pow(image_size//patch_size, 2)
        self.ishannet = nn.Linear(dim, 2)
        self.encoder = Encoder(image_size, patch_size, dim, head_encoder, h_dim, h_feedforward, depth_encoder)
        self.encoder_to_decoder = nn.Linear(dim, 3*patch_size*patch_size)
        self.decoder = Decoder(3*patch_size*patch_size, h_dim, 3*patch_size*patch_size, h_feedforward, patch_size, depth_decoder, head_decoder)
        self.ClS_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 3*patch_size*patch_size))
        self.pos = Positionembedding(dim)

    def forward(self, x, mask, num_mask):
        pos_emb = self.pos(x)  # all position
        x = x[:, mask == 1, :]
        x = self.projector(x)
        encoder_emb = pos_emb[:, mask == 1, :]  # unmasked position
        x = x+encoder_emb  # add absolute position
        B, N, C = x.shape
        self.ClS_token.data = self.ClS_token.data.expand(B, 1, self.dim).clone()
        x = torch.cat([self.ClS_token, x], dim=1)
        x = self.encoder(x)
        ishan_pre = x[:, 0, :]
        x = self.encoder_to_decoder(x)

        decoder_emb = pos_emb[:, mask == 1]
        decoder_emb = decoder_emb[:, :, 0]
        decoder_emb = decoder_emb.unsqueeze(2)
        decoder_emb = decoder_emb.expand(x.size(0), decoder_emb.size(1), 3*self.patch_size*self.patch_size).clone()
        masked_emb = pos_emb[:, mask == 0]
        masked_emb = masked_emb[:, :, 0]
        masked_emb = masked_emb.unsqueeze(2)
        masked_emb = masked_emb.expand(x.size(0), masked_emb.size(1), 3*self.patch_size*self.patch_size).clone()
        self.mask_token.data = self.mask_token.data.expand_as(masked_emb).clone()
        x = x[:, 1:, :]

        x = torch.cat([x+decoder_emb, self.mask_token+masked_emb], dim=1)
        x = x[:, -num_mask:, :]
        x = self.decoder(x)
        ishan_pre = self.ishannet(ishan_pre)
        return x, ishan_pre

if __name__ == '__main__':
    mae = MaskedAutoEncoder(dim=10, h_dim=20, h_feedforward=20, head_encoder=1, head_decoder=1, depth_encoder=1, depth_decoder=1, image_size=24, patch_size=12)
    mask= Mask(0.5)
    m, num_mask = mask.mask(4)
    x = torch.randn(2, 3, 24, 24)
    pe = PatchEmb(24, 12)
    x = pe.decompose(x)
    x = x.view(2, -1, 3*12*12)
    y, target = mae(x, m, num_mask)
    print(target.size())

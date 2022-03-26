import torch
import torch.nn as nn
from githubvit import vitmodel
from transformer import Transformer

class autoencoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, decoder_dim, num_classes=2, mlp_dim=1):
        super(autoencoder, self).__init__()
        self.encoder = vitmodel(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
        self.encoder_to_decoder = nn.Linear(dim, decoder_dim)
        self.encoder_to_decoderbn = nn.LayerNorm(decoder_dim)
        self.decoder = Transformer(dim=decoder_dim, heads=8, depth=12, h_dim=1024, h_forward=1024)
        self.decoder_to_output = nn.Linear(decoder_dim, 3 * 16 * 16)

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_to_decoder(x)
        x = self.encoder_to_decoderbn(x)
        x = self.decoder(x)
        x = self.decoder_to_output(x)
        return x

if __name__ == '__main__':
    ae = autoencoder(image_size=224, patch_size=16, dim=768, depth=12, heads=12, decoder_dim=512)
    x = torch.rand((1, 3, 224, 224))
    x = ae(x)
    print(x.size())
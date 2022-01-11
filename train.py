import torch
import torch.optim as optim
from MaskAutoEncoder import MaskedAutoEncoder
import torch.nn.functional as F
from Dataload import MaskAutoEncoderSet
from utils.vitmask import Mask
from utils.PatchGeneration import PatchEmb
from torch.utils.data import DataLoader


epoch = 100
patch_size = 16
image_size = 224
batch_size = 64
dim = 768
hidden_dim = 3072
heads_encoder = 12
heads_decoder = 12
depth_encoder = 8
depth_decoder = 12
traindataset = DataLoader(MaskAutoEncoderSet(), batch_size=batch_size, shuffle=True, drop_last=True)
testdataset = DataLoader(MaskAutoEncoderSet(eval=True), batch_size=2, shuffle=True, drop_last=True)
mae = MaskedAutoEncoder(dim=dim, h_dim=hidden_dim, h_feedforward=hidden_dim, head_encoder=heads_encoder, head_decoder=heads_decoder, depth_encoder=depth_encoder, depth_decoder=depth_decoder, image_size=image_size, patch_size=patch_size)
mask= Mask(0.75)
num_patch = pow(image_size//patch_size, 2)
pe = PatchEmb(image_size, patch_size)
optimizer = optim.SGD(mae.parameters(), lr=0.0001)
sumloss = 0.0
for i in range(epoch):
    for data in traindataset:
        x, y = data
        m, num_mask = mask.mask(num_patch)
        x = pe.decompose(x)
        x = x.view(x.size(0), -1, 3*patch_size*patch_size)
        target = x[:, m == 0, :]
        pre, ishan_pre = mae(x, m, num_mask)
        optimizer.zero_grad()
        loss1 = F.mse_loss(pre, target)
        loss2 = F.cross_entropy(ishan_pre, y)
        loss = (loss2+loss1)/batch_size
        sumloss += loss.item()
        loss.backward()
        optimizer.step()
    with open("Data/trainlog", 'a', encoding='utf-8') as f:
        f.write("epoch:{}".format(i) + '\t' + 'loss:{}'.format(sumloss))
torch.save(mae.state_dict(), "mae_dim={0}_he={1}_hd={2}_de={3}_dd={4}.pth".format(dim, heads_encoder, heads_decoder, depth_encoder, depth_decoder))

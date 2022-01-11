import torch
import torch.nn


class PatchEmb:
    def __init__(self, image_size, patch_size):
        super(PatchEmb, self).__init__()
        self.patch_size = patch_size
        self.image_size = image_size

    def decompose(self, x):
        patchlist = []
        for i in x.split(self.patch_size, dim=-2):
            for j in i.split(self.patch_size, dim=-1):
                patchlist.append(j.unsqueeze(1))
        patchlist = torch.cat(patchlist, dim=1)
        return patchlist

if __name__ == '__main__':
    pe = PatchEmb(224, 112)
    x = torch.zeros(2, 3, 224, 224)
    li = pe.decompose(x)
    print(li.size())
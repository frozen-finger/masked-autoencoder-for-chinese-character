import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MaskAutoEncoderSet(Dataset):
    def __init__(self, eval=False):
        super(MaskAutoEncoderSet, self).__init__()
        self.ishan_target = []
        self.fontlist = []
        self.eval = eval
        if not eval:
            with open("Data/Fontimage/train/train.txt", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                fontpath, target = line.strip('\n').split('\t')
                self.fontlist.append(fontpath)
                self.ishan_target.append(int(target))
        else:
            with open("Data/Fontimage/test/test.txt", encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                fontpath, target = line.strip('\n').split('\t')
                self.fontlist.append(fontpath)
                self.ishan_target.append((target))

    def __getitem__(self, item):
        path = self.fontlist[item]
        im = Image.open("Data/" + path)
        im = transforms.ToTensor()(im)
        target = torch.tensor(self.ishan_target[item], dtype=torch.long)
        return im, target

    def __len__(self):
        return len(self.fontlist)


if __name__ == '__main__':
    dataset = MaskAutoEncoderSet()
    im, target = dataset.__getitem__(1600)
    print(im.size())
    print(target)

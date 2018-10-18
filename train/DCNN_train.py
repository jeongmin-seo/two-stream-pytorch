from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.representation_loader as data_loader
from network.dynamic_k_max import DynamicKMaxPooling
from network.DCNN import DCNN
import numpy as np
import visdom
import os

batch_size = 32
data_root = "/home/jm/hdd/representation/split1"
text_root = "/home/jm/Two-stream_data/HMDB51"

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.group1= nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3)
        )

    def forward(self, _input):
        _input = _input.unsqueeze_(1)
        out = self.group1(_input)
        return out

def train_1epoch(data, _label, _d, _model):
    _input = _d(data, 2)
    _out = _model(_input)

    print(_out.size())

if __name__=="__main__":

    loader = data_loader.RepresentationLoader(batch_size, 8, data_root, text_root, 1, 19)
    train_loader = loader.run()
    d = DynamicKMaxPooling(8, 4)
    model = Conv()

    for i, (dat, label, dat_path) in enumerate(train_loader):
        train_1epoch(dat, label, d, model)

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.representation_loader as data_loader
from network.DCNN import DCNN
import numpy as np
import visdom
import os

batch_size = 32
data_root = "/home/jm/hdd/representation/split1"
text_root = "/home/jm/Two-stream_data/HMDB51"
if __name__=="__main__":

    loader = data_loader.RepresentationLoader(batch_size, 8, data_root, text_root, 1)
    train_loader = loader.run()

    for i, (dat, label, dat_path) in enumerate(train_loader):
        print(dat.size())
        print(len(os.listdir(dat_path[0])))


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
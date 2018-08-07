import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from network import Net

def train(_model, _train_loader):
    _model.train()
    for batch_idx, (data, target) in enumerate(_train_loader):

def test(_model):
    _model.eval()

def main():

    train_loader = DataLoader()

    model = Net().cuda(device=0)
    for epoch in range(1, 101):
        train(model)
        test(model)

if __name__ == '__main__':
    main()


#  A  A
# (‘ㅅ‘=)
# J.M.Seo

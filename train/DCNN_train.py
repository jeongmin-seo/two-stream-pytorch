from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.representation_loader as data_loader
from network.dynamic_k_max import DynamicKMaxPooling
from util.util import accuracy, frame2_video_level_accuracy, save_best_model
from network.DCNN import DCNN
import numpy as np
import visdom
import os

batch_size = 32
nb_epoch = 10000
data_root = "/home/jm/hdd/representation/split1"
text_root = "/home/jm/Two-stream_data/HMDB51"

class Conv(nn.Module):
    def __init__(self, batch_size):
        super(Conv, self).__init__()
        self.batch_size = batch_size
        self.dynamic_k_maxpool1 = DynamicKMaxPooling(5, 3)
        self.dynamic_k_maxpool2 = DynamicKMaxPooling(3, 3)
        """
        self.group1= nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        """
        self.conv1 = nn.Sequential(
            nn.Conv1d(2000, 1400, 3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1400, 1000, 3, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1000, 500, 3, padding=0),
            nn.ReLU()
        )
        self.fc = nn.Linear(1500, 51)


    def forward(self, _input):
        # out = self.dynamic_k_maxpool(_input, 0)
        out = self.conv1(_input)
        out = self.dynamic_k_maxpool1(out, 1)
        out = self.conv2(out)
        out = self.dynamic_k_maxpool2(out, 2)
        out = self.conv3(out)
        out = out.view(self.batch_size, -1)

        out = self.fc(out)

        return out

def train_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _model.train()
    for i, (dat, label, vid_root) in enumerate(_train_loader):
        label = label.cuda()
        input_var = Variable(dat).cuda()
        target_var = Variable(label).cuda().long()

        output = _model(input_var)

        loss = _loss_func(output, target_var)
        loss_list.append(loss)
        accuracy_list.append(accuracy(output.data, label))

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

    return float(sum(accuracy_list) / len(accuracy_list)), float(sum(loss_list) / len(loss_list)), _model

if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader = data_loader.RepresentationLoader(batch_size, 8, data_root, text_root, 1, 19)
    train_loader = loader.run()

    model = Conv(batch_size)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=2e-4)

    model = model.to(device)
    for epoch in range(1, nb_epoch +1):
    # for i, (dat, label, dat_path) in enumerate(train_loader):
        train_acc, train_loss, model = train_1epoch(model, train_loader, optimizer, criterion, epoch, nb_epoch)

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
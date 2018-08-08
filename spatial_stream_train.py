import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from network import Net
from util import accuracy, AverageMeter

import hmdb51

data_root = "/home/jm/Two-stream_data/HMDB51/preprocess/frames"
txt_root = "/home/jm/Two-stream_data/HMDB51"
batch_size = 32


def train_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    _model.train()
    for i, (data, label, video_name) in enumerate(_train_loader):

        label = label.cuda(async=True)
        input_var = Variable(data).cuda()
        target_var = Variable(label).cuda()

        output = _model(input_var)
        loss = _loss_func(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
        # print('Top1:', prec1, 'Top5:', prec5)

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

def val_1epoch(_model, _val_loader):
    _model.eval()

def main():
    training_set = hmdb51.HMDB(data_root, txt_root, 'train', 1,
                                 transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Scale([224,224]),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()
                                 ]))
    validation_set = hmdb51.HMDB(data_root, txt_root, 'test', 1,
                                transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Scale([224,224]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                ]))
    train_loader = DataLoader(dataset=training_set,
                              batch_size=batch_size,
                              shuffle=True
                              )
    val_loader = DataLoader(dataset=validation_set,
                            batch_size=batch_size,
                            shuffle=False)

    model = Net().cuda(device=0)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9)
    for epoch in range(1, 101):
        train_1epoch(model, train_loader, optimizer, criterion, epoch, 100)
        # val_1epoch(model, val_loader)

if __name__ == '__main__':
    main()


#  A  A
# (‘ㅅ‘=)
# J.M.Seo

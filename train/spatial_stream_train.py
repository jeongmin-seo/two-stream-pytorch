import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network.network import Net
from util import accuracy, frame2_video_level_accuracy, save_best_model
import pickle
import visdom
import numpy as np
import os

import data_loader.spatial_dataloader as data_loader

data_root = "/home/jeongmin/workspace/data/HMDB51/frames"
txt_root = "/home/jeongmin/workspace/data/HMDB51"
model_path = "/home/jeongmin/workspace/github/two-stream-pytorch/spatial_model"
batch_size = 128
nb_epoch = 10000


def train_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _model.train()
    for i, (data, label) in enumerate(_train_loader):

        label = label.cuda(async=True)
        input_var = Variable(data).cuda()
        target_var = Variable(label).cuda().long()

        output = _model(input_var)
        loss = _loss_func(output, target_var)
        loss_list.append(loss)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, label)  # , topk=(1, 5))
        accuracy_list.append(prec1)
        # print('Top1:', prec1)

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

    return float(sum(accuracy_list)/len(accuracy_list)), float(sum(loss_list)/len(loss_list)), _model


def val_1epoch(_model, _val_loader, _criterion, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][validation stage]'.format(_epoch, _nb_epochs))

    dic_video_level_preds = {}
    dic_video_level_targets = {}
    _model.eval()
    for i, (video, data, label) in enumerate(_val_loader):

        label = label.cuda(async=True)
        input_var = Variable(data, volatile=True).cuda(async=True)
        target_var = Variable(label, volatile=True).cuda(async=True)

        # compute output
        output = _model(input_var)

        # Calculate video level prediction
        preds = output.data.cpu().numpy()
        nb_data = preds.shape[0]
        for j in range(nb_data):
            videoName = video[j]
            if videoName not in dic_video_level_preds.keys():
                dic_video_level_preds[videoName] = preds[j, :]
                dic_video_level_targets[videoName] = target_var[j]
            else:
                dic_video_level_preds[videoName] += preds[j, :]
                # dic_video_level_targets[videoName] += target_var[j, :]

    video_acc, video_loss = frame2_video_level_accuracy(dic_video_level_preds, dic_video_level_targets, _criterion)

    return video_acc, video_loss, dic_video_level_preds


def main():

    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    loader = data_loader.Spatial_DataLoader(BATCH_SIZE=batch_size, num_workers=8,
                                            path=data_root, txt_path=txt_root, split_num=1)

    train_loader, test_loader, test_video = loader.run()
    # model = Net(channel=3).cuda(device=0)
    model = torch.load(os.path.join(model_path, '161_epoch_best_model.pth'))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
    cur_best_acc = 0
    for epoch in range(162, nb_epoch+1):
        train_acc, train_loss, model = train_1epoch(model, train_loader, optimizer, criterion, epoch, nb_epoch)
        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = val_1epoch(model, test_loader, criterion, epoch, nb_epoch)
        print("Validation Accuracy:", val_acc, "Validation Loss:", val_loss)

        # lr scheduler
        scheduler.step(val_loss)

        is_best = val_acc > cur_best_acc
        if is_best:
            cur_best_acc = val_acc
            with open('./spatial_pred/spatial_video_preds.pickle','wb') as f:
                pickle.dump(video_level_pred, f)
            f.close()

        vis.line(X=np.asarray([epoch]), Y=np.asarray([train_loss]),
                 win=loss_plot, update="append", name='Train Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([train_acc]),
                 win=acc_plot, update="append", name="Train Accuracy")
        vis.line(X=np.asarray([epoch]), Y=np.asarray([val_loss]),
                 win=loss_plot, update="append", name='Validation Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([val_acc]),
                 win=acc_plot, update="append", name="Validation Accuracy")
        save_best_model(is_best, model, model_path, epoch)


if __name__ == '__main__':
    main()


#  A  A
# (‘ㅅ‘=)
# J.M.Seo

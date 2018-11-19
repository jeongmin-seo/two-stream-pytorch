import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import visdom
import os
import sys
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import data_loader.temporal_dataloader as data_loader
from network.network import *
from util.util import accuracy, frame2_video_level_accuracy, save_best_model, str2bool, make_save_dir

###################################
#     argument parser setting     #
###################################
parser = argparse.ArgumentParser(description='Pytorch Action Recognition temporal stream')
parser.add_argument('--data_root', type=str, help="set data root")
parser.add_argument('--text_root', type=str, help="set train test split file root")
parser.add_argument('--split_num', type=int, choices=[1,2,3], help='set train test split number')
parser.add_argument('--save_root', type=str, default="./")
parser.add_argument('--model', type=str,
                    choices=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152'], help='set model')
parser.add_argument('--pretrained', type=str, default='true')
parser.add_argument('--img_size', type=int, default=224, help="set train image size")
parser.add_argument('--stack_size', type=int, choices=[5,10], help="set stack size")
parser.add_argument('--learning_rate','--lr', type=float, default=0.001, help="set train learning rate")
parser.add_argument('--batch_size', '--bs', type=int, default=16, help="set batch size")
parser.add_argument('--epoch', type=int, default=10000, help="set train epoch number")


def set_model(model_name, stack_size, pretrained):
    if model_name == 'resnet18':
        select_model = resnet18(pretrained=pretrained, channel=stack_size*2)
    elif model_name == 'resnet34':
        select_model = resnet34(pretrained=pretrained, channel=stack_size*2)
    elif model_name == 'resnet50':
        select_model = resnet50(pretrained=pretrained, channel=stack_size*2)
    elif model_name == 'resnet101':
        select_model = resnet101(pretrained=pretrained, channel=stack_size*2)
    elif model_name == 'resnet152':
        select_model = resnet152(pretrained=pretrained, channel=stack_size*2)
    else:
        raise ValueError

    return select_model

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
        prec1 = accuracy(output.data, label)
        accuracy_list.append(prec1)

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

    video_acc, video_loss = frame2_video_level_accuracy(dic_video_level_preds, dic_video_level_targets, _criterion)

    return video_acc, video_loss, dic_video_level_preds


def main():
    global args
    args = parser.parse_args()

    save_path = os.path.join(args.save_root, "temporal_" + args.model)
    make_save_dir(save_path)

    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    loader = data_loader.MotionDataLoader(img_size = args.img_size, batch_size=args.batch_size,
                                          num_workers=8, in_channel=args.stack_size,path=args.data_root,
                                          txt_path=args.text_root, split_num=args.split_num)
    train_loader, test_loader, test_video = loader.run()

    model = set_model(args.model, args.stack_size, str2bool(args.pretrained))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda(device=device)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)


    cur_best_acc = 0
    for epoch in range(1, args.epoch+1):
        train_acc, train_loss, model = train_1epoch(model, train_loader, optimizer, criterion, epoch, args.epoch)
        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = val_1epoch(model, test_loader, criterion, epoch, args.epoch)
        print("Validation Accuracy:", val_acc, "Validation Loss:", val_loss)

        # lr scheduler
        scheduler.step(val_loss)

        is_best = val_acc > cur_best_acc
        if is_best:
            cur_best_acc = val_acc
            with open(os.path.join(save_path,'temporal_video_preds.pickle'), 'wb') as f:
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
        save_best_model(is_best, model, save_path, epoch)


if __name__ == '__main__':
    main()


#  A  A
# (‘ㅅ‘=)
# J.M.Seo

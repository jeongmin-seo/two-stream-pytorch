import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import visdom
import os
import sys
import copy
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import data_loader.spatial_cube_dataloader as data_loader
import network.resnet_3d as resnet
import network.resnext_3d as resnext
from util.util import accuracy, frame2_video_level_accuracy, save_best_model, str2bool, make_save_dir

###################################
#     argument parser setting     #
###################################
parser = argparse.ArgumentParser(description='Pytorch Action Recognition temporal stream')
parser.add_argument('--data_root', type=str, help="set data root")
parser.add_argument('--text_root', type=str, help="set train test split file root")
parser.add_argument('--split_num', type=int, choices=[1,2,3], help='set train test split number')
parser.add_argument('--save_root', type=str, default="./")
parser.add_argument('--model', type=str, choices=['resnext-101','resnet-101','resnet-152'], help='set model')
parser.add_argument('--train_type', type=str, choices=['tsn'],
                    default='tsn', help='set train type tsn or two-stream')
parser.add_argument('--modality', choices=['rgb', 'flow'], default='rgb', help="select data modality")
parser.add_argument('--pretrained', type=str, default='true')
parser.add_argument('--pretrained_root', type=str, help="3d resnet pretrained model root")
parser.add_argument('--img_size', type=int, default=224, help="set train image size")
parser.add_argument('--stack_size', type=int, choices=[16,32,64], default=16, help="set stack size")
parser.add_argument('--learning_rate','--lr', type=float, default=0.001, help="set train learning rate")
parser.add_argument('--batch_size', '--bs', type=int, default=4, help="set batch size")
parser.add_argument('--epoch', type=int, default=10000, help="set train epoch number")



def set_model(model_name, img_size, stack_size, pretrained, pretrained_root):

    if model_name == 'resnet-101':
        select_model = resnet.resnet101(sample_size=img_size, sample_duration=stack_size)
    elif model_name == 'resnet-152':
        select_model = resnet.resnet152(sample_size=img_size, sample_duration=stack_size)
    elif model_name == 'resnext-101':
        select_model = resnext.resnet101(sample_size=img_size, sample_duration=stack_size)

    else:
        raise ValueError

    if pretrained:
        if model_name=='resnext-101' and stack_size == 64:
            state_dict = torch.load(os.path.join(pretrained_root, model_name+'-64f-kinetics.pth'))
        else:
            state_dict = torch.load(os.path.join(pretrained_root, model_name + '-kinetics.pth'))
        state_dict = refine_state_dict(state_dict)
        select_model.load_state_dict(state_dict['state_dict'])

    in_feature = select_model.fc.in_features
    select_model.fc = nn.Linear(in_feature, 51)

    return select_model

def refine_state_dict(state_dict):
    # this code is pretrained model load code
    new_state_dict = copy.deepcopy(state_dict)
    for key in state_dict['state_dict'].keys():
        new_key = key.split('.', 1)[1]
        new_state_dict['state_dict'][new_key] = state_dict['state_dict'][key]
        del new_state_dict['state_dict'][key]

    return new_state_dict


def train_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _model.train()
    for i, (data, label) in enumerate(_train_loader):
        label = label.cuda()
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

    return float(sum(accuracy_list) / len(accuracy_list)), float(sum(loss_list) / len(loss_list)), _model

def train_tsn_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _model.train()

    for i, (data, label) in enumerate(_train_loader):
        label = label.cuda()
        # input_var = Variable(data).cuda()
        target_var = Variable(label).cuda().long()

        for i, dat in enumerate(data):
            input_var = Variable(dat).cuda()
            if i == 0:
                output = _model(input_var)
            else:
                output = output + _model(input_var)
        loss = _loss_func(output/3, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, label)

        loss_list.append(loss)
        accuracy_list.append(float(prec))

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

    return float(sum(accuracy_list) / len(accuracy_list)), float(sum(loss_list) / len(loss_list)), _model

def val_1epoch(_model, _val_loader, _criterion, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][validation stage]'.format(_epoch, _nb_epochs))

    dic_video_level_preds = {}
    dic_video_level_targets = {}
    _model.eval()
    for i, (video, data, label) in enumerate(_val_loader):

        label = label.cuda(async=True)
        with torch.no_grad():
            input_var = Variable(data).cuda(async=True)
            target_var = Variable(label).cuda(async=True)

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

    save_path = os.path.join(args.save_root, "3d_spatial" + args.model)
    make_save_dir(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()
    train_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    train_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    loader = data_loader.CubeDataLoader(img_size = args.img_size,
                                        batch_size=args.batch_size,
                                        num_workers=8,
                                        in_channel=args.stack_size,
                                        path=args.data_root,
                                        txt_path=args.text_root,
                                        split_num=args.split_num,
                                        train_type=args.train_type,
                                        modality=args.modality)
    train_loader, test_loader, test_video = loader.run()


    # state_dict = torch.load(os.path.join(save_path, "resnet-101-kinetics-hmdb51_split1.pth"))
    model = set_model(args.model, args.img_size, args.stack_size, str2bool(args.pretrained), args.pretrained_root)
    # model = model.cuda()

    if args.modality == 'flow':
        model.conv1 = nn.Conv3d(
                2,
                64,
                kernel_size=7,
                stride=(1, 2, 2),
                padding=(3, 3, 3),
                bias=False)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.5,0.999), lr=args.learning_rate)
    model = model.to(device)
    cur_best_acc = 0

    # multi gpu
    # model = torch.nn.DataParallel(model, device_ids=[0,1])

    for epoch in range(1, args.epoch+1):
        if args.train_type == 'tsn':
            train_acc, train_loss, model = \
                train_tsn_1epoch(model, train_loader, optimizer, criterion, epoch, args.epoch)
        elif args.train_type == 'two-stream':
            train_acc, train_loss, model = \
                train_1epoch(model, train_loader, optimizer, criterion, epoch, args.epoch)

        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = val_1epoch(model, test_loader, criterion, epoch, args.epoch)
        print("Validation Accuracy:", val_acc, "Validation Loss:", val_loss)

        is_best = val_acc > cur_best_acc
        if is_best:
            cur_best_acc = val_acc
            with open(os.path.join(save_path,'3d_spatial_preds.pickle'), 'wb') as f:
                pickle.dump(video_level_pred, f)
            f.close()

        vis.line(X=np.asarray([epoch]), Y=np.asarray([train_loss]),
                 win=train_loss_plot, update="append", name='Train Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([train_acc]),
                 win=train_acc_plot, update="append", name="Train Accuracy")
        vis.line(X=np.asarray([epoch]), Y=np.asarray([val_loss]),
                 win=val_loss_plot, update="append", name='Validation Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([val_acc]),
                 win=val_acc_plot, update="append", name="Validation Accuracy")
        save_best_model(is_best, model, save_path, epoch)


#Test network
if __name__ == '__main__':
    main()

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
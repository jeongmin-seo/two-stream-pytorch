import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import visdom
import numpy as np
import os
import pickle
import copy
from network import resnet_3d
import data_loader.spatial_cube_dataloader as data_loader
from util.util import accuracy, frame2_video_level_accuracy, save_best_model


# experimental parameters
data_root = "/home/jm/Two-stream_data/HMDB51/original/frames"
txt_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/workspace/two-stream-pytorch/spatial_cube_model"
batch_size = 4
nb_epoch = 10000
L = 16
n_class = 51

# C3D Model
class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        #init.xavier_normal(self.group1.state_dict()['weight'])
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group2.state_dict()['weight'])
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group3.state_dict()['weight'])
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group4.state_dict()['weight'])
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group5.state_dict()['weight'])

        self.fc1 = nn.Sequential(
            nn.Linear(86528, 2048),               #
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 51), #101
            nn.Softmax()
        )

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4
            # self.group5
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        out = self._features(x)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self._classifier(out)
        return self.fc3(out)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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

        loss = 0
        prec = 0
        for dat in data:
            input_var = Variable(dat).cuda()
            output = _model(input_var)
            loss += _loss_func(output, target_var)

            # measure accuracy and record loss
            prec += accuracy(output.data, label)

        loss_list.append(loss)
        accuracy_list.append(float(prec)/3)

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    loader = data_loader.CubeDataLoader(BATCH_SIZE=batch_size, num_workers=8, in_channel=L,
                                        path=data_root, txt_path=txt_root, split_num=1,mode='spatial')

    train_loader, test_loader, test_video = loader.run()

    # model = resnet_3d.resnet18(sample_size=112, sample_duration=L)
    # model = resnet_3d.resnet34(sample_size=112, sample_duration=32)
    state_dict = torch.load(os.path.join(save_path, "resnet-101-kinetics-hmdb51_split1.pth"))
    model = resnet_3d.resnet101(sample_size=108, sample_duration=L)

    new_state_dict = copy.deepcopy(state_dict)
    for key in state_dict['state_dict'].keys():
        new_key = key.split('.', 1)[1]
        new_state_dict['state_dict'][new_key] = state_dict['state_dict'][key]
        del new_state_dict['state_dict'][key]
    del state_dict

    model.load_state_dict(new_state_dict['state_dict'])
    parameters = resnet_3d.get_fine_tuning_parameters(model, 50)

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), betas=(0.5,0.999), lr=2e-4)
    optimizer = torch.optim.SGD(parameters, lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    model = model.to(device)
    cur_best_acc = 0
    for epoch in range(nb_epoch+1):
        # train_acc, train_loss, model = train_1epoch(model, train_loader, optimizer, criterion, epoch, nb_epoch)
        train_acc, train_loss, model = train_tsn_1epoch(model, train_loader, optimizer, criterion, epoch, nb_epoch)
        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = val_1epoch(model, test_loader, criterion, epoch, nb_epoch)
        print("Validation Accuracy:", val_acc, "Validation Loss:", val_loss)

        # lr scheduler
        scheduler.step(val_loss)

        is_best = val_acc > cur_best_acc
        if is_best:
            cur_best_acc = val_acc
            with open('../spatial_cube_model/spatial_cube_preds.pickle', 'wb') as f:
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


#Test network
if __name__ == '__main__':
    main()

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
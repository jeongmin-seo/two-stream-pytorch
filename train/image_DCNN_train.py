from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.image_DCNN_loader as data_loader
import torchvision.models as models
from network.dynamic_k_max import DynamicKMaxPooling
import numpy as np
import visdom
import pickle
from util.util import accuracy, frame2_video_level_accuracy, save_best_model

batch_size = 4
nb_epoch = 10000
max_frame_num = 1063
data_root = "/home/jm/hdd/representation/split1"
text_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/hdd/dynamic_k_max_model"


class ConvTemporal(nn.Module):
    def __init__(self, batch_size):
        super(ConvTemporal, self).__init__()
        self.batch_size = batch_size
        self.dynamic_k_maxpool1 = DynamicKMaxPooling(16, 5)

        self.conv1 = nn.Sequential(
            nn.Conv1d(2000, 1700, 3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(1700, 1400, 3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(1400, 1000, 3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1000, 700, 3),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(700, 400, 3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(6400, 4000),
            nn.Linear(4000, 2000),
            nn.Linear(2000, 51)
        )


    def forward(self, _input):
        out = self.conv1(_input)
        out = self.dynamic_k_maxpool1(out, 1)
        out = self.conv2(out)
        out = self.dynamic_k_maxpool1(out, 2)
        out = self.conv3(out)
        out = self.dynamic_k_maxpool1(out, 3)
        out = self.conv4(out)
        out = self.dynamic_k_maxpool1(out, 4)
        out = self.conv5(out)
        out = self.dynamic_k_maxpool1(out, 5)
        out = out.view(out.size()[0], -1)

        out = self.fc(out)

        return out


def train_1epoch(_img_feature_extract_model, _dcnn_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _img_feature_extract_model.train()
    _dcnn_model.train()
    for i, (dat, label, video_name) in enumerate(_train_loader):
        label = label.cuda()
        # input_var = Variable(dat).cuda()
        target_var = Variable(label).cuda().long()

        print(dat.size())
        for batch_idx in range(batch_size):
            for frame_idx in range(max_frame_num):
                output = _img_feature_extract_model(dat[batch_idx, :, frame_idx, :].unsqueeze_(0).cuda())
                print(output.size())

        # output = _model(input_var)

        loss = _loss_func(output, target_var)
        loss_list.append(loss.data)
        accuracy_list.append(accuracy(output.data, label))

        # compute gradient and do SGD step
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()

    return float(sum(accuracy_list) / len(accuracy_list)), float(sum(loss_list) / len(loss_list)), _img_feature_extract_model, _dcnn_model


def validation_1epoch(_img_feature_extract_model, _dcnn_model, _val_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][validation stage]'.format(_epoch, _nb_epochs))

    dic_video_level_preds = {}
    dic_video_level_targets = {}
    _img_feature_extract_model.eval()
    _dcnn_model.eval()
    for i, (dat, label, video_name) in enumerate(_val_loader):
        label = label.cuda(async=True)
        with torch.no_grad():
            input_var = Variable(dat).cuda(async=True)
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

    video_acc, video_loss = frame2_video_level_accuracy(dic_video_level_preds, dic_video_level_targets, _loss_func)

    return video_acc, video_loss, dic_video_level_preds


if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()
    train_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    train_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    loader = data_loader.DCNNLoader(batch_size=batch_size, num_workers=4, path=data_root,
                                    txt_path=text_root, split_num=1, max_frame_num=max_frame_num)
    train_loader, val_loader = loader.run()

    feature_extract_model = models.resnet34(pretrained=True)
    action_decision_model = ConvTemporal(batch_size)

    feature_extract_model.to(device)
    action_decision_model.to(device)
    parameters = list(feature_extract_model.parameters()) + list(action_decision_model.parameters())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(parameters, betas=(0.5, 0.999), lr=2e-4)

    cur_best_acc = 0
    for epoch in range(1, nb_epoch +1):
        train_acc, train_loss, feature_extract_model, action_decision_model = train_1epoch(feature_extract_model,
                                                                                           action_decision_model,
                                                                                           train_loader,
                                                                                           optimizer,
                                                                                           criterion,
                                                                                           epoch,
                                                                                           nb_epoch)
        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = validation_1epoch(feature_extract_model, action_decision_model,
                                                                val_loader, optimizer, criterion, epoch, nb_epoch)
        print("Validation Accuracy:", val_acc, "Validation Loss:", val_loss)

        is_best = val_acc > cur_best_acc
        if is_best:
            cur_best_acc = val_acc
            with open('../spatial_cube_model/spatial_cube_preds.pickle', 'wb') as f:
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
        # save_best_model(is_best, model, save_path, epoch)

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
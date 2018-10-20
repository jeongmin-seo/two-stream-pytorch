from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.dynamic_k_maxpool_loader as data_loader
from network.dynamic_k_max import DynamicKMaxPooling
import numpy as np
import visdom
import pickle
from util.util import accuracy, frame2_video_level_accuracy, save_best_model

batch_size =32
nb_epoch = 10000
data_root = "/home/jm/hdd/representation/split1"
text_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/hdd/dynamic_k_max_model"

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
        out = out.view(out.size()[0], -1)

        out = self.fc(out)

        return out

def train_1epoch(_model, _train_loader, _optimizer, _loss_func, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][training stage]'.format(_epoch, _nb_epochs))

    accuracy_list = []
    loss_list = []
    _model.train()
    for i, (dat, label) in enumerate(_train_loader):
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

def validation_1epoch(_model, _val_loader, _optimizer, _criterion, _epoch, _nb_epochs):
    print('==> Epoch:[{0}/{1}][validation stage]'.format(_epoch, _nb_epochs))

    dic_video_level_preds = {}
    dic_video_level_targets = {}
    _model.eval()
    for i, (video, dat, label) in enumerate(_val_loader):
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

    video_acc, video_loss = frame2_video_level_accuracy(dic_video_level_preds, dic_video_level_targets, _criterion)


    return video_acc, video_loss, dic_video_level_preds

if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()
    train_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    train_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    val_acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    loader = data_loader.RepresentationLoader(batch_size, 8, data_root, text_root, 1, 19)
    train_loader, val_loader = loader.run()

    model = Conv(batch_size)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=2e-4)

    model = model.to(device)
    cur_best_acc = 0
    for epoch in range(1, nb_epoch +1):
        train_acc, train_loss, model = train_1epoch(model, train_loader, optimizer, criterion, epoch, nb_epoch)
        print("Train Accuacy:", train_acc, "Train Loss:", train_loss)
        val_acc, val_loss, video_level_pred = validation_1epoch(model, val_loader, optimizer, criterion, epoch, nb_epoch)
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
        save_best_model(is_best, model, save_path, epoch)

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 96 output channels, 7x7 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 96, 7, padding='same')
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, padding='same')
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv4 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv5 = nn.Conv2d(512, 512, 3, padding='same')
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv1_bn(x), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(self.conv2_bn(x), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(self.conv5(x), (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(nn.Linear(x.size, 4096), p=0.5, training=self.training)
        x = F.dropout(nn.Linear(x.size, 2048), p=0.5, training=self.training)
        x = F.Variable(nn.Linear(x.size, 51))

        return F.softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
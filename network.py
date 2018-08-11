import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 96 output channels, 7x7 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 96, 7, padding=1)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv1_bn(x), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(self.conv2_bn(x), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(self.conv5(x), (2, 2))

        # x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(nn.Linear(self.num_flat_features(x), 4096)(x), p=0.5, training=self.training)
        x = F.dropout(nn.Linear(4096, 2048)(x), p=0.5, training=self.training)
        x = F.Variable(nn.Linear(2048, 51)(x))

        return F.softmax(x)
    """

    def __init__(self, channel):
        super(Net, self).__init__()
        # 3 input image channel, 96 output channels, 7x7 square convolution
        # kernel
        self._in_channel = channel
        self.fully_size = 4608
        self.conv = nn.Sequential(
            nn.Conv2d(self._in_channel, 96, 7, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(96, 256, 5, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, 3),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fully_conneted = nn.Sequential(
            nn.Linear(4608, 4096),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.Dropout(),
            nn.Linear(2048, 51),
            nn.Softmax()
        )

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv(x)
        # print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))

        return self.fully_conneted(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#  A  A
# (‘ㅅ‘=)
# J.M.Seo
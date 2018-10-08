# this is dynamic convolutional neural network for action recognition.
import torch.nn as nn
from network.dynamic_k_max import DynamicKMaxPooling

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1,32,7),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = DynamicKMaxPooling(16, 2)(out)
        print(out.size())


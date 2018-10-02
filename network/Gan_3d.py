import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 6
        self.init_channel = 4
        self.l1 = nn.Sequential(nn.Linear(5000, 512 * self.init_channel * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 256, kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=(3, 5, 5), stride=(1,2,2)),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=(3, 9, 9), stride=(1,2,2)),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 3, kernel_size=(5, 8, 8), stride=(1,2,2)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_channel, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_class=0, img_channel=3):
        super(Discriminator, self).__init__()
        self.n_class = n_class + 1
        self.image_channel = img_channel

        self.model = nn.Sequential(
            nn.Conv3d(self.image_channel, 64, kernel_size=(5,7,7), stride=(1,2,2)),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=(3,7,7), stride=(1,2,2)),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=(3,5,5), stride=(1,2,2)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3),
            nn.ReLU(),

            nn.Conv3d(256, 512, kernel_size=3),
            nn.ReLU()
        )

        # The height and width of downsampled image
        # self.adv_layer = nn.Sequential(nn.Linear(73728, 1),
        #                                nn.Sigmoid())
        self.classifer_layer = nn.Sequential(nn.Linear(73728, self.n_class)) # +1 is fake label relate

    def forward(self, img_cube):
        out = self.model(img_cube)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        validity = self.adv_layer(out)

        return validity

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
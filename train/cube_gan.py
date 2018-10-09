from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.spatial_cube_dataloader as data_loader
from network.autoencoder_2d import UnNormalize
# import data_loader.spatial_cube_dataloader as data_loader
# from util import save_best_model
import numpy as np
import visdom
import os

cuda = True if torch.cuda.is_available() else False

# experimental parameters
# data_root = "/home/jm/Two-stream_data/HMDB51/original/frames"
data_root = "/home/jm/Two-stream_data/HMDB51/original/flow"
txt_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/hdd/temporal_gan_model"
batch_size = 32
nb_epoch = 10000
L = 16
n_class = 51

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 1# 6
        self.init_channel = 4
        # self.l1 = nn.Sequential(nn.Linear(5000, 512 * self.init_channel * self.init_size * self.init_size))
        self.l1 = nn.Sequential(nn.Linear(2048, 512 * self.init_channel * self.init_size * self.init_size))
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 256, kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=(3, 5, 5), stride=(1,2,2)),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=(3, 7, 7), stride=(1,2,2)),
            nn.ReLU(True),

            # nn.ConvTranspose3d(64, 3, kernel_size=(5, 8, 8), stride=(1,2,2)),
            nn.ConvTranspose3d(64, 2, kernel_size=(5, 8, 8), stride=(1,2,2)),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_channel, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(5,7,7), stride=(1,2,2)),
            # nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(1,2,2)),
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
        self.adv_layer = nn.Sequential(# nn.Linear(73728, 1),
                                       nn.Linear(2048, 1),
                                       nn.Sigmoid())
        self.classifer_layer = nn.Sequential(nn.Linear(73728, n_class+1)) # +1 is fake label relate

    def forward(self, img_cube):
        out = self.model(img_cube)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)

        return validity

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # loader = data_loader.SpatialCubeDataLoader(BATCH_SIZE=batch_size, num_workers=8, in_channel=L,
    #                                            path=data_root, txt_path=txt_root, split_num=1)
    # train_loader, test_loader, test_video = loader.run()

    loader = data_loader.CubeDataLoader(BATCH_SIZE=batch_size, num_workers=8, in_channel=L,
                                        path=data_root, txt_path=txt_root, split_num=1, mode="temporal")
    train_loader, test_loader, test_video = loader.run()

    # visdom init
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    # init discriminator
    discriminator = Discriminator()
    discriminator.train()
    discriminator = discriminator.to(device)

    # init generator
    generator = Generator()
    generator.train()
    generator = generator.to(device)

    # set optimizer
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))

    unnorm = UnNormalize(mean=[0.5,], std=[0.5,])

    prev_err = None
    for epoch in range(nb_epoch):
        generator_error = []
        discriminator_error = []
        for i, (data, label) in enumerate(train_loader):

            # discriminator train
            discriminator.zero_grad()

            input_var = Variable(data).cuda()
            target = Variable(torch.ones(input_var.size()[0])).cuda()
            # target = Variable(label).cuda()
            output = discriminator(input_var)
            err_real = criterion(output, target)

            # noise = Variable(torch.randn(input_var.size()[0], latent_Vector, 1, 1)).cuda()
            # noise = Variable(torch.randn(input_var.size()[0], 5000)).cuda()
            noise = Variable(torch.randn(input_var.size()[0], 2048)).cuda()
            fake = generator(noise)
            # target = Variable(torch.ones(input_var.size()[0])*51).cuda()   # fake data label is 51
            target = Variable(torch.zeros(input_var.size()[0])).cuda()
            output = discriminator(fake.detach())
            err_fake = criterion(output, target)

            errD = err_real + err_fake
            errD.backward()
            optimizerD.step()

            # generator train
            generator.zero_grad()
            # target = Variable(torch.ones(input_var.size()[0])*51).cuda()
            target = Variable(torch.zeros(input_var.size()[0])).cuda()
            output = discriminator(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, nb_epoch, i, len(train_loader), errD.data[0], errG.data[0]))
            generator_error.append(errG.data[0])
            discriminator_error.append(errD.data[0])

            X = unnorm(fake[0,0,0,:,:])
            Y = unnorm(fake[0,1,0,:,:])
            vis.image(X, win="fake image X", opts=dict(size=(68, 68)))
            vis.image(Y, win="fake image Y", opts=dict(size=(68, 68)))

        generator_model_err = np.mean(np.asarray(generator_error))
        discriminator_model_err = np.mean(np.asarray(discriminator_error))

        if epoch % 20 == 0 and epoch != 0 :
            torch.save(generator, os.path.join(save_path, '%d_epoch.pkl' %epoch))
        """
        if not prev_err or prev_err > generator_model_err:
            save_best_model(True, generator, save_path, epoch)
            prev_err = generator_model_err
        """
        vis.line(X=np.asarray([epoch]), Y=np.asarray([generator_error]),
                 win=loss_plot, update="append", name='Train G Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([discriminator_model_err]),
                 win=acc_plot, update="append", name="Train D Loss")

#  A  A
# (‘ㅅ‘=)
# J.M.Seo

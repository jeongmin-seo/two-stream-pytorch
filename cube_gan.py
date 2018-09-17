from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import spatial_cube_dataloader as data_loader

cuda = True if torch.cuda.is_available() else False

# experimental parameters
data_root = "/home/jm/Two-stream_data/HMDB51/original/frames"
txt_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/workspace/two-stream-pytorch/spatial_cube_model"
batch_size = 1
nb_epoch = 10000
L = 16
n_class = 51

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.init_size = opt.img_size // 4
        self.init_size = 3
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        self.l1 = nn.Sequential(nn.Linear(5000, 128 * self.init_size * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose3d(128, 512, kernel_size=3),
            nn.ReLU(),

            nn.ConvTranspose3d(512, 256, kernel_size=3),
            nn.ReLU(),
            # nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.ConvTranspose3d(256, 256, kernel_size=3),
            nn.ReLU(),


            nn.ConvTranspose3d(256, 128, kernel_size=3),
            nn.ReLU(),
            # nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.ConvTranspose3d(128, 64, kernel_size=3),
            nn.ReLU(),
            # nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),

            nn.ConvTranspose3d(64, 3, kernel_size=5),

            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(5,7,7), stride=(1,2,2)),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3,7,7), stride=(1,2,2)),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3,5,5), stride=(1,2,2)),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=3),
            nn.ReLU()
        )

        # The height and width of downsampled image
        self.adv_layer = nn.Sequential(nn.Linear(73728, 1),
                                       nn.Sigmoid())

    def forward(self, img_cube):
        out = self.model(img_cube)
        print(out.size())
        out = out.view(out.size(0), -1)
        print(out.size())
        validity = self.adv_layer(out)

        return validity

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = data_loader.SpatialCubeDataLoader(BATCH_SIZE=batch_size, num_workers=8, in_channel=L,
                                               path=data_root, txt_path=txt_root, split_num=1)
    train_loader, test_loader, test_video = loader.run()

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
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for i, (data, label) in enumerate(train_loader):

        # discriminator train
        discriminator.zero_grad()

        input_var = Variable(data).cuda()
        target = Variable(torch.ones(input_var.size()[0])).cuda()
        output = discriminator(input_var)
        err_real = criterion(output, target)

        # noise = Variable(torch.randn(input_var.size()[0], latent_Vector, 1, 1)).cuda()
        noise = Variable(torch.randn(input_var.size()[0], 5000)).cuda()
        fake = generator(noise)
        target = Variable(torch.zeros(input_var.size()[0]))
        output = discriminator(fake.detach())
        err_fake = criterion(output, target)

        errD = err_real + err_fake
        errD.backward()
        optimizerD.step()

        # generator train
        generator.zero_grad()
        target = Variable(torch.ones(input_var.size()[0]))
        output = discriminator(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(train_loader), errD.data[0], errG.data[0]))

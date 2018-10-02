from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import data_loader.JHMDB_loader as data_loader
from network.Gan_3d import Discriminator, Generator
from util import save_best_model
import numpy as np
import visdom

cuda = True if torch.cuda.is_available() else False

# experimental parameters
data_root = "/home/jm/hdd/JHMDB/background_extract_image"
txt_root = "/home/jm/hdd/JHMDB/train_split_1.txt"
save_path = "/home/jm/workspace/two-stream-pytorch/cube_gan_model_jhmdb"
batch_size = 4
nb_epoch = 10000
L = 8
n_class = 21

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = data_loader.JHMDBLoader(batch_size=batch_size, num_workers=8, in_channel=L,
                                     path=data_root, txt_path=txt_root)
    train_loader = loader.load()

    # visdom init
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    # init discriminator
    discriminator = Discriminator(img_channel=1)
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

    prev_err = None
    for epoch in range(nb_epoch):
        generator_error = []
        discriminator_error = []
        for i, (data, label) in enumerate(train_loader):

            # discriminator train
            discriminator.zero_grad()

            input_var = Variable(data).cuda()
            target = Variable(label).cuda()
            output = discriminator(input_var)
            err_real = criterion(output, target)

            # noise = Variable(torch.randn(input_var.size()[0], latent_Vector, 1, 1)).cuda()
            noise = Variable(torch.randn(input_var.size()[0], 5000)).cuda()
            fake = generator(noise)
            target = Variable(torch.ones(input_var.size()[0])*51).cuda()   # fake data label is 51
            output = discriminator(fake.detach())
            err_fake = criterion(output, target)

            errD = err_real + err_fake
            errD.backward()
            optimizerD.step()

            # generator train
            generator.zero_grad()
            target = Variable(torch.ones(input_var.size()[0])*51).cuda()
            output = discriminator(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, nb_epoch, i, len(train_loader), errD.data[0], errG.data[0]))
            generator_error.append(errG.data[0])
            discriminator_error.append(errD.data[0])

        generator_model_err = np.mean(np.asarray(generator_error))
        discriminator_model_err = np.mean(np.asarray(discriminator_error))

        if not prev_err or prev_err > generator_model_err:
            save_best_model(True, generator, save_path, epoch)
            prev_err = generator_model_err

        vis.line(X=np.asarray([epoch]), Y=np.asarray([generator_model_err]),
                 win=loss_plot, update="append", name='Train G Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([discriminator_model_err]),
                 win=acc_plot, update="append", name="Train D Loss")

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
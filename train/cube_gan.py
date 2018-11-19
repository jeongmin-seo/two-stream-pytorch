import torch
from torch.autograd import Variable
import visdom
import os
import sys
import numpy as np
import argparse
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import data_loader.gan_loader as data_loader
from network.Gan_3d import *
from util.util import make_save_dir

###################################
#     argument parser setting     #
###################################
parser = argparse.ArgumentParser(description='Pytorch Action Recognition temporal stream')
parser.add_argument('--data_root', type=str, help="set data root")
parser.add_argument('--save_root', type=str, default="./")
parser.add_argument('--img_size', type=int, default=68, help="set train image size")
parser.add_argument('--stack_size', type=int, default=16, help="set stack size")
parser.add_argument('--learning_rate','--lr', type=float, default=0.001, help="set train learning rate")
parser.add_argument('--batch_size', '--bs', type=int, default=4, help="set batch size")
parser.add_argument('--epoch', type=int, default=10000, help="set train epoch number")

n_class = 51




if __name__ == "__main__":

    global args
    args = parser.parse_args()

    save_path = os.path.join(args.save_root, "3d_gan")
    make_save_dir(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = data_loader.CubeDataLoader(img_size=args.img_size,
                                        batch_size=args.batch_size,
                                        num_workers=8,
                                        in_channel=args.stack_size,
                                        path=args.data_root)
    train_loader = loader.train()

    # visdom init
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    # init discriminator
    discriminator = Discriminator(n_class=n_class, img_channel=2)
    discriminator.train()
    discriminator = discriminator.to(device)

    # init generator
    generator = Generator()
    generator.train()
    generator = generator.to(device)

    # set optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))

    invTrans = transforms.Compose([#transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.,], std=[1/0.5,]),
                                   transforms.Normalize(mean=[-0.5,], std=[1., ])])

    prev_err = None
    for epoch in range(1, args.epoch+1):
        generator_error = []
        discriminator_error = []
        for i, (data, label) in enumerate(train_loader):

            # discriminator train
            discriminator.zero_grad()

            input_var = Variable(data).cuda()
            target = Variable(label).cuda()
            # target = Variable(label).cuda()
            output = discriminator(input_var)
            err_real = criterion(output, target)

            # noise = Variable(torch.randn(input_var.size()[0], latent_Vector, 1, 1)).cuda()
            # noise = Variable(torch.randn(input_var.size()[0], 5000)).cuda()
            noise = Variable(torch.randn(input_var.size()[0], 2048)).cuda()
            fake = generator(noise)
            target = Variable(torch.ones(input_var.size()[0])*51).cuda().long()   # fake data label is 51
            # target = Variable(torch.zeros(input_var.size()[0])).cuda()
            output = discriminator(fake.detach())
            err_fake = criterion(output, target)

            errD = err_real + err_fake
            errD.backward()
            optimizerD.step()

            # generator train
            generator.zero_grad()
            # target = Variable(torch.ones(input_var.size()[0])).cuda()
            output = discriminator(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, args.epoch, i, len(train_loader), errD.data[0], errG.data[0]))
            generator_error.append(errG.data[0])
            discriminator_error.append(errD.data[0])

            fakeX = invTrans(fake[0,0,0:3,:,:])   # because, pytorch transform recognize only 3-dimension data as an image
            fakeY = invTrans(fake[0,1,0:3,:,:])   # because, pytorch transform recognize only 3-dimension data as an image
            vis.image(fakeX[0,:,:], win="fake image X", opts=dict(size=(68, 68)))
            vis.image(fakeY[0,:,:], win="fake image Y", opts=dict(size=(68, 68)))

            realX = invTrans(input_var[0,0,0:3,:,:])   # because, pytorch transform recognize only 3-dimension data as an image
            realY = invTrans(input_var[0,1,0:3,:,:])    # because, pytorch transform recognize only 3-dimension data as an image
            vis.image(realX[0,:,:], win="real image X", opts=dict(size=(68, 68)))
            vis.image(realY[0,:,:], win="real image Y", opts=dict(size=(68, 68)))

        generator_model_err = np.mean(np.asarray(generator_error))
        discriminator_model_err = np.mean(np.asarray(discriminator_error))

        if epoch % 20 == 0 and epoch != 0 :
            torch.save(generator, os.path.join(save_path, '%d_epoch.pkl' %epoch))

        vis.line(X=np.asarray([epoch]), Y=np.asarray([generator_error]),
                 win=loss_plot, update="append", name='Train G Loss')
        vis.line(X=np.asarray([epoch]), Y=np.asarray([discriminator_model_err]),
                 win=acc_plot, update="append", name="Train D Loss")

#  A  A
# (‘ㅅ‘=)
# J.M.Seo

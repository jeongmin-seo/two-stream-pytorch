###################################
# requirment library              #
###################################
# import data_loader.spatial_dataloader as data_loader
import data_loader.extract_representation_loader as data_loader
from torch.autograd import Variable
import visdom
import numpy as np
import torch
import torch.nn as nn
import os
import re
from network.autoencoder_2d import Encoder, Decoder, UnNormalize
import argparse
from util.custom_error import WrongSelectError
import torchvision.models as models

###################################
# select mode                     #
###################################
parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--data_root', metavar='DIR', default='/home/jm/Two-stream_data/HMDB51/original/frames',
                    help='path to datset root')
parser.add_argument('--text_root', '-t', default='/home/jeongmin/workspace/data/HMDB51',
                    help='path to train test split text files root')
parser.add_argument('--split_number', default=1, help='select split number', choices=[1, 2, 3])
parser.add_argument('--save_path', '-s', default='/home/jm/hdd/frame_ae_model')
parser.add_argument('--mode', '-m', default='test', choices=["train", "test"])
parser.add_argument('--batch_size', '-b', default=1)
parser.add_argument('--epoch', '-e', default=10000)

###################################
# requirment variable             #
###################################
# data_root = "/home/jm/Two-stream_data/HMDB51/original/frames"
# txt_root = "/home/jm/Two-stream_data/HMDB51"
# save_path = "/home/jm/workspace/two-stream-pytorch/frame_ae_model"
# batch_size = 32
# nb_epoch = 10000


def train():
    loader = data_loader.SpatialDataLoader(BATCH_SIZE=args.batch_size, num_workers=8, path=args.data_root,
                                           txt_path=args.text_root, split_num=args.split_number, is_ae=True)
    train_loader, test_loader, test_video = loader.run()

    # visdom init
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    # acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init encoder
    encoder = Encoder(_batch_size=args.batch_size)
    encoder.train()
    encoder = encoder.to(device)

    # init decoder
    decoder = Decoder(_batch_size=args.batch_size)
    decoder.train()
    decoder = decoder.to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    # set optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=0.00002, betas=(0.5, 0.999))

    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prev_loss = None
    for epoch in range(args.epoch):
        loss_list = []
        for i, (data, label) in enumerate(train_loader):
            # label = label.cuda(async=True)
            input_var = Variable(data, volatile=True).cuda(async=True)
            # target_var = Variable(label, volatile=True).cuda(async=True)

            latent = encoder(input_var)
            output = decoder(latent)
            loss = criterion(output, input_var)
            loss_list.append(loss.data[0])

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            origin_img = unnorm(input_var[0])
            recon_img = unnorm(output[0])
            vis.image(origin_img, win="input image", opts=dict(size=(64, 64)))
            vis.image(recon_img, win="output image")

        cur_loss = np.mean(np.asarray(loss_list))
        if not prev_loss:
            prev_loss = cur_loss

        if prev_loss > cur_loss:
            # save_best_model(True, model, save_path, epoch)
            torch.save([encoder, decoder], os.path.join(args.save_path, 'best_ae.pkl'))
            prev_loss = cur_loss

        vis.line(X=np.asarray([epoch]), Y=np.asarray([cur_loss]),
                 win=loss_plot, update="append", name='Train Loss')


def test():
    result_path = "/home/jm/hdd/resnet_feature"
    loader = data_loader.RepresentationLoader(batch_size=args.batch_size, num_workers=8, path=args.data_root)
    represetation_loader = loader.run()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = models.resnet101(pretrained=True)
    encoder.to(device)
    encoder.eval()
    """
    # model load
    model = torch.load("/home/jm/hdd/frame_ae_model/best_ae_779epoch.pkl")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init encoder
    encoder = Encoder(_batch_size=args.batch_size)
    encoder.train()
    encoder = encoder.to(device)
    print(model[0])
    encoder.load_state_dict(model[0].state_dict())

    nomalize = nn.Sequential(
        nn.Sigmoid()
    )
    """
    for i, (data, data_path) in enumerate(represetation_loader):
        input_var = Variable(data, volatile=True).cuda(async=True)
        latent = encoder(input_var)
        # norm_output = nomalize(latent)

        print(latent.size())
        dir_name = re.split("[/]+", data_path[0])
        save_path = os.path.join(result_path, dir_name[-3])

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        save_path = os.path.join(save_path, dir_name[-2])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        save_name = os.path.join(save_path, dir_name[-1])
        save_name = save_name.replace('.jpg', '.npy')

        np.save(save_name, latent.data.cpu().numpy())


if __name__ == "__main__":

    global args
    args = parser.parse_args()
    """
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        raise WrongSelectError("No Named " + args.mode + "in args")
    """
    test()

#  A  A
# (‘ㅅ‘=)
# J.M.Seo

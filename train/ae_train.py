###################################
# requirment library              #
###################################
import data_loader.spatial_dataloader as data_loader
from torch.autograd import Variable
import visdom
import numpy as np
import torch
import torch.nn as nn
import os
from network.autoencoder_2d import Encoder, Decoder, UnNormalize

###################################
# requirment variable             #
###################################
data_root = "/home/jm/Two-stream_data/HMDB51/original/frames"
txt_root = "/home/jm/Two-stream_data/HMDB51"
save_path = "/home/jm/workspace/two-stream-pytorch/frame_ae_model"
batch_size = 32
nb_epoch = 10000



if __name__=="__main__":

    loader = data_loader.Spatial_DataLoader(BATCH_SIZE=batch_size, num_workers=8,
                                            path=data_root, txt_path=txt_root, split_num=1,is_ae=True)
    train_loader, test_loader, test_video = loader.run()

    # visdom init
    vis = visdom.Visdom()
    loss_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))
    # acc_plot = vis.line(X=np.asarray([0]), Y=np.asarray([0]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init encoder
    encoder = Encoder(_batch_size=batch_size)
    encoder.train()
    encoder = encoder.to(device)

    # init decoder
    decoder = Decoder(_batch_size=batch_size)
    decoder.train()
    decoder = decoder.to(device)

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    # set optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=0.00002, betas=(0.5, 0.999))

    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    prev_loss = None
    for epoch in range(nb_epoch):
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
            vis.image(origin_img, win="input image",opts=dict(size=(64,64)))
            vis.image(recon_img, win="output image")

        cur_loss = np.mean(np.asarray(loss_list))
        if not prev_loss:
            prev_loss = cur_loss

        if prev_loss > cur_loss:
            # save_best_model(True, model, save_path, epoch)
            torch.save([encoder,decoder], os.path.join(save_path, 'best_ae.pkl'))
            prev_err = cur_loss

        vis.line(X=np.asarray([epoch]), Y=np.asarray([cur_loss]),
                 win=loss_plot, update="append", name='Train Loss')



#  A  A
# (‘ㅅ‘=)
# J.M.Seo
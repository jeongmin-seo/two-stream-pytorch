import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, _batch_size, channel=3):
        super(Encoder, self).__init__()

        self._channel = channel
        self.batch_size = _batch_size

        self.conv = nn.Sequential(
            nn.Conv2d(self._channel, 96, 7, stride=2),
            nn.ReLU(),

            nn.Conv2d(96, 256, 5, stride=2),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3),
        )

        self.enc_linear = nn.Sequential(
            nn.Linear(8192,5000),
            nn.Linear(5000, 2000)
        )



    def forward(self, image):
        out = self.conv(image)
        out = out.view(self.batch_size,-1)
        out = self.enc_linear(out)

        # z = flatten1 + flatten2
        return out

class Decoder(nn.Module):

    def __init__(self, _batch_size, channel=3):
        super(Decoder, self).__init__()

        self._channel = channel
        self.batch_size = _batch_size
        self.dec_linear = nn.Sequential(
            nn.Linear(2000, 5000),
            nn.Linear(5000, 8192)
        )
        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(512, 512, 3),

            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, stride=2),

            nn.ReLU(True),
            nn.ConvTranspose2d(256, 96, 5, stride=2),

            nn.ReLU(True),
            nn.ConvTranspose2d(96, self._channel, 8, stride=2),

            nn.Tanh()
        )

    def forward(self, z):
        out = self.dec_linear(z)
        out = out.view(self.batch_size, 512, 4, 4)
        out = self.deconv(out)

        # z = flatten1 + flatten2
        return out

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

#  A  A
# (‘ㅅ‘=)
# J.M.Seo
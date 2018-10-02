import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, z):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

        )

        self.classification = nn.Sequential(
            nn.Linear(128*ds_size**2, n_class + 1),
            nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validaty = self.classfication(out)

        return validaty
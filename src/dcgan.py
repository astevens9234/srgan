"""Implimentation of Deep Convolutional Generative Adversarial Network.
c.f. https://arxiv.org/abs/1511.06434
"""

import datetime as dt
import os

import torch
import torchvision

from torch import nn

from .gan_util import Accumulator, update_D, update_G, extract_zip

# TODO: Logging/Warnings/cProfile/etc


class DCGAN(nn.Module):
    """Base Class Deep Convolutional Generative Adversarial Network"""

    def __init__(self, resize=(64, 64), batch_size=64, data_dir="../data"):
        super(DCGAN, self).__init()
        self.transform = torchvision.transforms.Compose(
            torchvision.transforms.Resize(resize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        )
        self.data_dir = torchvision.datasets.DatasetFolder(data_dir)
        self.data_iter = torch.utils.data.DataLoader(
            self.data_dir, batch_size=batch_size, shuffle=True
        )


class G_block(nn.Module):
    """Generator Block."""

    def __init__(
        self, out_channels, in_channels=3, kernel_size=4, stride=2, padding=1, **kwargs
    ):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))


class D_block(nn.Module):
    """Discriminator Block."""

    def __init__(
        self,
        out_channels,
        in_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        alpha=0.2,
        **kwargs
    ):
        super(D_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))


# projects z (noise) into n_G*8 channels, then halves the channel each time
# convolution layer generates output, doubling size
n_G = 64
net_G = nn.Sequential(
    G_block(in_channels=100, out_channels=n_G * 8, stride=1, padding=0),
    G_block(in_channels=n_G * 8, out_channels=n_G * 4),
    G_block(in_channels=n_G * 4, out_channels=n_G * 2),
    G_block(in_channels=n_G * 2, out_channels=n_G),
    nn.ConvTranspose2d(
        in_channels=n_G, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False
    ),
    nn.Tanh(),
)

# Verify 100 dimensional input results in 64x64 output...
if False:
    x = torch.zeros((1, 100, 1, 1))
    print(net_G(x).shape)


# Opposite the generator, with convolution output=1 to make single prediction.
n_D = 64
net_D = nn.Sequential(
    D_block(n_D),
    D_block(in_channels=n_D, out_channels=n_D * 2),
    D_block(in_channels=n_D * 2, out_channels=n_D * 4),
    D_block(in_channels=n_D * 4, out_channels=n_D * 8),
    nn.Conv2d(in_channels=n_D * 8, out_channels=1, kernel_size=4, bias=False),
)

# Verify that the 64x64 input results in a single value
if False:
    x = torch.zeros((1, 3, 64, 64))
    print(net_D(x).shape)


# TODO: Move this into the utils
def training(net_D, net_G, device, data_iter, lr=0.005, num_epochs=10, latent_dim=100):
    loss = nn.BCEWithLogitsLoss(reduction="sum")

    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)

    net_D, net_G = net_D.to(device), net_G.to(device)
    grid = {"lr": lr, "betas": [0.5, 0.999]}

    trainer_D = torch.optim.Adam(net_D.parameters(), **grid)
    trainer_G = torch.optim.Adam(net_G.parameters(), **grid)

    for _ in range(0, num_epochs):
        metric = Accumulator(3)
        for X, _ in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1))
            (
                X,
                Z,
            ) = X.to(
                device
            ), Z.to(device)
            metric.add(
                update_D(X, Z, net_D, net_G, loss, trainer_D),
                update_G(Z, net_D, net_G, loss, trainer_G),
                batch_size,
            )
            loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
            print("loss_D: {}, loss_G {}".format(loss_D, loss_G))

    time = dt.datetime.now().strftime("%d-%m-%Y-%H-%M")
    torch.save(net_G.state_dict(), "./models/dcgan-netg-{}.params".format(time))
    torch.save(net_D.state_dict(), "./models/dcgan-netd-{}.params".format(time))


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main(batch_size=256, url="http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip"):
    """Training DCGAN."""
    extract_zip(url=url, folder="../data")
    data = torchvision.datasets.ImageFolder("../data")
    # NOTE: Abstract out Compose params to accomodate different input resolutions?
    data.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )
    data_iter = torch.utils.data.DataLoader(data, batch_size, shuffle=True)

    training(net_D, net_G, device, data_iter)


main()

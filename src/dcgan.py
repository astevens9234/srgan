"""Implimentation of Deep Convolutional Generative Adversarial Network."""

import torch
import torchvision

from torch import nn


class DCGAN(nn.Module):
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

    def forward(X, self):
        return self.activation(self.batch_norm(self.conv2d(X)))

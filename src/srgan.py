"""Pytorch implimentation of SRGAN model, described here:
c.f. <https://arxiv.org/pdf/1609.04802>
"""

from torch import nn


class SRGAN(nn.Module):

    def __init__(self, kernel=9, n_map=64, stride=1):
        """Super Resolution Generative Adversarial Network.
        
        Args:
        """
        super(SRGAN, self).__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class G_Network(nn.module):

    def __init__(self):
        """Generator Network"""
        super(G_Network, self).__init__()
        
        self.conv2d_tran = nn.ConvTranspose2d()
        # NOTE: Paper calls for ParametricReLU, which appears to be absent from the pytorch API.
        #       Will leave as a leaky for now while scaffolding - FIXME.
        self.leaky_relu = nn.LeakyReLU()

        self.r_block = (
            nn.Sequential(
                Residual_Block(),
                Residual_Block(),
                Residual_Block(),
                Residual_Block(),
                Residual_Block(),
            ),
        )

        self.conv2d_tran

    def residual_block(self, ): ...

class D_Network(nn.Module):

    def __init__(self):
        """Discriminator Network. Eight convolutional layers followed by two dense layers."""
        super(D_Network, self).__init__()
        self.conv2d_tran = nn.ConvTranspose2d()
        self.leaky_relu = nn.LeakyReLU()
        self.d_block = nn.Sequential(
            self.discriminator_block(kernel_size=3, in_channels=64, out_channels=128, stride=1),
            self.discriminator_block(kernel_size=3, in_channels=128, out_channels=256, stride=2),
            self.discriminator_block(kernel_size=3, in_channels=256, out_channels=256, stride=1),
            self.discriminator_block(kernel_size=3, in_channels=256, out_channels=512, stride=2),
            self.discriminator_block(kernel_size=3, in_channels=512, out_channels=512, stride=1),
            self.discriminator_block(kernel_size=3, in_channels=512, out_channels=1024, stride=2)
        )
        self.dense = nn.Sequential(
            nn.LazyLinear(out_features=1024, bias=True),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1, bias=True),
            nn.Sigmoid()
        )

    def discriminator_block(self, in_channels=64, out_channels=64, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(kernel_size, in_channels, out_channels, stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )
        

class Residual_Block(nn.Module):

    def __init__(self):
        """Residual Block. Two Convolution layers with 3x3 kernels, 64 feature maps, followed by batch-norm and parametric ReLU."""
        super(Residual_Block, self).__init__()
        self.block = nn.Sequential(

        )

    ...


class ContentLoss(nn.Module):

    def __init__(self):
        """19 Layer VGG loss purposed at: <https://arxiv.org/pdf/1409.1556>"""
        super(ContentLoss, self).__init__()

    def forward(self):
        raise NotImplementedError


class AdversarialLoss(nn.Module):

    def __init__(self):
        """"""
        super(AdversarialLoss, self).__init__()

    def forward(self):
        raise NotImplementedError


class PerceptualLoss(nn.Module):

    def __init__(self):
        """Weighted sum of Content Loss & Adversarial Loss."""
        super(PerceptualLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss()

    def forward(self):
        return self.content_loss + ((10**-3) * self.adversarial_loss)

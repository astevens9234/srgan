"""Pytorch implimentation of SRGAN model, described here:
c.f. <https://arxiv.org/pdf/1609.04802>

Training data can be found here: <https://storage.googleapis.com/openimages/web/index.html>
"""

import torch

from torch import nn
from torchvision import models, transforms


class SRGAN(nn.Module):

    def __init__(self):
        """Super Resolution Generative Adversarial Network.

        Args:
        """
        super(SRGAN, self).__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class G_Network(nn.Module):

    def __init__(self):
        """Generator Network"""
        super(G_Network, self).__init__()

        self.conv2d_tran = nn.ConvTranspose2d()
        # NOTE: Paper calls for ParametricReLU, which appears to be absent from the pytorch API.
        #       Will leave as a leaky for now while scaffolding - FIXME.
        #       UPDATE a leaky relu w/ learnable slope is a parametric relu :)
        self.leaky_relu = nn.LeakyReLU()

        self.r_block = (
            nn.Sequential(
                self.residual_block(),
                self.residual_block(),
                self.residual_block(),
                self.residual_block(),
                self.residual_block(),
            ),
        )

        self.conv2d_tran

    def residual_block(
        self,
    ): ...


class D_Network(nn.Module):

    def __init__(self):
        """Discriminator Network. Eight convolutional layers followed by two dense layers."""
        super(D_Network, self).__init__()
        self.conv2d_tran = nn.ConvTranspose2d()
        self.leaky_relu = nn.LeakyReLU()
        self.d_block = nn.Sequential(
            self.discriminator_block(
                kernel_size=3, in_channels=64, out_channels=128, stride=1
            ),
            self.discriminator_block(
                kernel_size=3, in_channels=128, out_channels=256, stride=2
            ),
            self.discriminator_block(
                kernel_size=3, in_channels=256, out_channels=256, stride=1
            ),
            self.discriminator_block(
                kernel_size=3, in_channels=256, out_channels=512, stride=2
            ),
            self.discriminator_block(
                kernel_size=3, in_channels=512, out_channels=512, stride=1
            ),
            self.discriminator_block(
                kernel_size=3, in_channels=512, out_channels=1024, stride=2
            ),
        )
        self.dense = nn.Sequential(
            nn.LazyLinear(out_features=1024, bias=True),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def discriminator_block(
        self, in_channels=64, out_channels=64, kernel_size=3, stride=1
    ):
        return nn.Sequential(
            nn.Conv2d(kernel_size, in_channels, out_channels, stride),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
        )


class ContentLoss(nn.Module):
    """MSE Between feature maps"""

    def __init__(self, layer_index=20, device="cuda"):
        """19 Layer VGG loss purposed at: <https://arxiv.org/pdf/1409.1556>."""
        super(ContentLoss, self).__init__()

        device = torch.device(device)

        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval().to(device)
        self.layer_index = layer_index

        for param in self.vgg.parameters():
            param.requires_grad = False

        # NOTE: mean/std are stock values
        # TODO: calc these values from the training set
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to(device)

        self.mse_loss = nn.MSELoss()

    def forward(self, input_img, target_img):
        """
        Args:
            input_img (torch.Tensor): Generated image (B, C, H, W)
            target_img (torch.Tensor): Target (B, C, H, W)
        Returns:
            loss (torch.Tensor)
        """
        input_img = self.normalize(input_img)
        target_img = self.normalize(target_img)

        for i, layer in enumerate(self.vgg): # type: ignore
            input_img = layer(input_img)
            target_img = layer(target_img)

            if i == self.layer_index:
                return self.mse_loss(input_img, target_img)


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

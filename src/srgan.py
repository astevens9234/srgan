"""Pytorch implimentation of SRGAN model, described here:
c.f. <https://arxiv.org/pdf/1609.04802>
"""

import torch

from torch import nn
from torchvision import models, transforms, datasets


class SRGAN(nn.Module):

    def __init__(self):
        """Super Resolution Generative Adversarial Network."""
        super(SRGAN, self).__init__()

        self.generator = G_Network()
        self.discriminator = D_Network()


class G_Network(nn.Module):
    """Generator Module.
    - Expects Low Resolution (LR) image in input, ex [batch, 3, 16, 16],
    - downsampled from source [batch, 3, 64, 64] in the training loop.

    Output:
        Image in Super Resolution: [batch, 3, 64, 64]
    """

    def __init__(self):
        """Generator Network"""
        super(G_Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4
            ),
            nn.PReLU(),
        )

        resblockargs = {
            "in_channels": 64,
            "kernel_size": 3,
            "out_channels": 64,
            "stride": 1,
            "padding": 1,
        }

        self.net.add_module(
            name="Residual Blocks",
            module=nn.Sequential(
                ResidualBlock(**resblockargs),
                ResidualBlock(**resblockargs),
                ResidualBlock(**resblockargs),
                ResidualBlock(**resblockargs),
                ResidualBlock(**resblockargs),
            ),
        )

        self.net.add_module(name="Elementwise Sum Block", module=ElementSumBlock())

        self.net.add_module(
            name="Pixel Shuffler Block",
            module=nn.Sequential(
                PixelShuffleBlock(in_channels=64, out_channels=64*16),
                # PixelShuffleBlock(in_channels=16, out_channels=256),
                nn.Conv2d(
                    in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4
                ),
            ),
        )

    def forward(self, X):
        X = self.net[:2](X)
        skip_connection = X
        X = self.net[2](X)
        X = self.net[3](X)
        X = X + skip_connection
        X = self.net[4](X)
        return X


class ResidualBlock(nn.Module):
    """Wrapper for residual blocks, required to wrap nn.Sequential in nn.Module to perform elementwise sums."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, X):
        residual = X.clone()
        output = self.sequential(X)
        return output + residual


class ElementSumBlock(nn.Module):
    """Wrapper for the Element Sum Block. Adds initial residual to output"""

    def __init__(self):
        super(ElementSumBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=64),
        )

    def forward(self, X):
        output = self.sequential(X)
        return output


class PixelShuffleBlock(nn.Module):
    """Wrapper for Pixel Shuffle Blocks."""

    def __init__(self, in_channels, out_channels):
        super(PixelShuffleBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=4),
            nn.PReLU(),
        )

    def forward(self, X):
        return self.sequential(X)


class D_Network(nn.Module):
    """Discriminator Module.
    Expects input from the Generator network, which outputs [batch, 3, 64, 64]
    Outputs tensor [batch, 1]
    """

    def __init__(self):
        """Discriminator Network. Eight convolutional layers followed by two dense layers."""
        super(D_Network, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),              
            nn.LeakyReLU(negative_slope=0.02),                                                          
            DiscriminatorBlock(kernel_size=3, in_channels=64, out_channels=64, stride=2, padding=1),    
            DiscriminatorBlock(kernel_size=3, in_channels=64, out_channels=128, stride=1, padding=1),   
            DiscriminatorBlock(kernel_size=3, in_channels=128, out_channels=128, stride=2, padding=1),  
            DiscriminatorBlock(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=1),  
            DiscriminatorBlock(kernel_size=3, in_channels=256, out_channels=256, stride=2, padding=1),  
            DiscriminatorBlock(kernel_size=3, in_channels=256, out_channels=512, stride=1, padding=1),  
            DiscriminatorBlock(kernel_size=3, in_channels=512, out_channels=512, stride=2, padding=1),  
        )

        self.net.add_module(
            name="Dense Layer",
            module=nn.Sequential(
            nn.Linear(in_features=512*8*8, out_features=1028, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(in_features=1028, out_features=1, bias=True),
            nn.Sigmoid()
            )
        )

    def forward(self, X):
        X = self.net[:8](X)
        X = X.view(64, -1) # Flatten for the Dense layers
        X = self.net[9](X)
        return X


class DiscriminatorBlock(nn.Module):
    """Wrapper for discriminator blocks."""

    def __init__(
        self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
    ):
        super(DiscriminatorBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.02),
        )

    def forward(self, X):
        return self.net(X)


class ContentLoss(nn.Module):

    def __init__(self, layer_index=20):
        """19 Layer VGG loss purposed at: <https://arxiv.org/pdf/1409.1556>.
        Takes MSE Between Feature Maps.
        """
        super(ContentLoss, self).__init__()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval().to(device)
        self.layer_index = layer_index

        for param in self.vgg.parameters():
            param.requires_grad = False

        # NOTE: mean/std are stock values
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
            loss
        """
        input_img = self.normalize(input_img)
        target_img = self.normalize(target_img)

        for i, layer in enumerate(self.vgg):  # type: ignore
            input_img = layer(input_img)
            target_img = layer(target_img)

            if i == self.layer_index:
                return self.mse_loss(input_img, target_img)


class AdversarialLoss(nn.Module):

    def __init__(self):
        """Compute adversarial loss for the discriminator.
        Real image targes == 1, fake images == 0.

        Args:
            D_real (torch.Tensor): High Resolution image passed through the Discriminator Network
            G_I_LR (torch.Tensor): Low Resolution image passed through the Generator Network

        Returns:
            loss (torch.Tensor)
        """
        super(AdversarialLoss, self).__init__()

    def forward(self, D_real, G_I_LR):
        real_labels = torch.ones_like(D_real)
        loss_D_real = nn.BCELoss()(D_real, real_labels)
        D_fake = G_I_LR.detach()  # Detach to prevent backprop through G
        fake_labels = torch.zeros_like(D_fake)
        loss_D_fake = nn.BCELoss()(D_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake

        return loss_D


class PerceptualLoss(nn.Module):

    def __init__(self):
        """Weighted sum of Content Loss & Adversarial Loss."""
        super(PerceptualLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss()

    def forward(self, D_real, G_I_LR):
        al = self.adversarial_loss(D_real, G_I_LR)
        cl = self.content_loss()
        return cl + (10**-3 * al)

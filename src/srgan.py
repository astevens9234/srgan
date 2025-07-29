"""Pytorch implimentation of SRGAN model, described here:
c.f. <https://arxiv.org/pdf/1609.04802>
"""

import torch

from torch import nn
from torchvision import models, transforms, datasets


class SRGAN(nn.Module):

    def __init__(self, resize=(64, 64), batch_size=256, data_dir="../data"):
        """Super Resolution Generative Adversarial Network.

        Args:
            resize: Downsampling size
            batch_size: Number of images per batch
            data_dir: Location of data
            url: Download location, expects .zip file link.
        """
        super(SRGAN, self).__init__()

        data = datasets.ImageNet(data_dir, transform=transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(
            data, batch_size, shuffle=True
        )  # , num_workers=4)

        for X, y in data_loader:
            print("Shape of X [N, C, H, W]: ".format(X.shape))
            print(y.shape)

        self.generator = G_Network()
        self.discriminator = D_Network()


class G_Network(nn.Module):

    def __init__(self):
        """Generator Network"""
        super(G_Network, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConvTranspose2d(out_channels=64, kernel_size=9, stride=1), nn.PReLU()
        )

        self.skip_connection = nn.Identity()

        self.net.add_module(
            name="Residual Blocks",
            module=nn.Sequential(
                ResidualBlock(kernel_size=3, out_channels=64, stride=1),
                ResidualBlock(kernel_size=3, out_channels=64, stride=1),
                ResidualBlock(kernel_size=3, out_channels=64, stride=1),
                ResidualBlock(kernel_size=3, out_channels=64, stride=1),
                ResidualBlock(kernel_size=3, out_channels=64, stride=1),
            ),
        )

        self.net.add_module(
            name="Elementwise Sum Block",
            module=ElementSumBlock(skip_connection=self.skip_connection),
        )

        self.net.add_module(
            name="Pixel Shuffler Block",
            module=nn.Sequential(
                PixelShuffleBlock(),
                PixelShuffleBlock(),
                nn.LazyConv2d(kernel_size=9, out_channels=3, stride=1),
            ),
        )

    def forward(self, X):
        return self.net(X)


class ResidualBlock(nn.Module):
    """Wrapper for residual blocks, required to wrap nn.Sequential in nn.Module to perform elementwise sums."""

    def __init__(self, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyConv2d(
                out_channels=out_channels, kernel_size=kernel_size, stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
            nn.LazyConv2d(
                out_channels=out_channels, kernel_size=kernel_size, stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, X):
        residual = X
        output = self.sequential(X)
        return output + residual


class ElementSumBlock(nn.Module):
    """Wrapper for the Element Sum Block. Adds initial residual to output"""

    def __init__(self, skip_connection):
        super(ElementSumBlock, self).__init__()
        self.skip_connection = skip_connection
        self.sequential = nn.Sequential(
            nn.LazyConv2d(kernel_size=3, out_channels=64, stride=1),
            nn.BatchNorm2d(num_features=64),
        )

    def forward(self, X):
        output = self.sequential(X)
        return output + self.skip_connection


class PixelShuffleBlock(nn.Module):
    """Wrapper for Pixel Shuffle Blocks."""

    def __init__(self):
        super(PixelShuffleBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.LazyConv2d(kernel_size=3, out_channels=256, stride=1),
            nn.PixelShuffle(upscale_factor=3),
            nn.PixelShuffle(upscale_factor=3),
            nn.PReLU(),
        )

    def forward(self, X):
        return self.sequential(X)


class D_Network(nn.Module):

    def __init__(self):
        """Discriminator Network. Eight convolutional layers followed by two dense layers."""
        super(D_Network, self).__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(kernel_size=3, out_channels=64, stride=1),
            nn.LeakyReLU(negative_slope=0.02),
            DiscriminatorBlock(
                kernel_size=3, in_channels=64, out_channels=128, stride=1
            ),
            DiscriminatorBlock(
                kernel_size=3, in_channels=128, out_channels=256, stride=2
            ),
            DiscriminatorBlock(
                kernel_size=3, in_channels=256, out_channels=256, stride=1
            ),
            DiscriminatorBlock(
                kernel_size=3, in_channels=256, out_channels=512, stride=2
            ),
            DiscriminatorBlock(
                kernel_size=3, in_channels=512, out_channels=512, stride=1
            ),
            DiscriminatorBlock(
                kernel_size=3, in_channels=512, out_channels=1024, stride=2
            ),
            nn.LazyLinear(out_features=1024, bias=True),
            nn.LeakyReLU(negative_slope=0.02),
            nn.LazyLinear(out_features=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, X):
        return self.net(X)


class DiscriminatorBlock(nn.Module):
    """Wrapper for discriminator blocks."""

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1):
        super(DiscriminatorBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.02),
        )

    def forward(self, X):
        return self.net(X)


class ContentLoss(nn.Module):

    def __init__(self, layer_index=20, device="cuda"):
        """19 Layer VGG loss purposed at: <https://arxiv.org/pdf/1409.1556>.
        Takes MSE Between Feature Maps.
        """
        super(ContentLoss, self).__init__()

        device = torch.device(device)

        vgg = models.vgg19(pretrained=True).features
        self.vgg = vgg.eval().to(device)
        self.layer_index = layer_index

        for param in self.vgg.parameters():
            param.requires_grad = False

        # NOTE: mean/std are stock values
        # TODO: calc these values from the training set
        # NOTE: shouldn't normalization happen at input stage?
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

        for i, layer in enumerate(self.vgg):  # type: ignore
            input_img = layer(input_img)
            target_img = layer(target_img)

            if i == self.layer_index:
                return self.mse_loss(input_img, target_img)


class AdversarialLoss(nn.Module):

    def __init__(self, D_real, G_I_LR):
        """
        Args:
            D_real (torch.Tensor): High Resolution image passed through the Discriminator Network
            G_I_LR (torch.Tensor): Low Resolution image passed through the Generator Network

        Returns:
            loss (torch.Tensor)
        """
        super(AdversarialLoss, self).__init__()
        self.loss_d = self.compute_D_loss(D_real, G_I_LR)

    def compute_D_loss(self, D_real, G_I_LR) -> torch.Tensor:
        """
        Compute adversarial loss for the discriminator.
        Real image targes == 1, fake images == 0.

        Args:
            D_real (torch.Tensor): High Resolution image passed through the Discriminator Network
            G_I_LR (torch.Tensor): Low Resolution image passed through the Generator Network

        Returns:
            loss_D: Adversarial Loss
        """

        real_labels = torch.ones_like(D_real)
        loss_D_real = nn.BCELoss()(D_real, real_labels)

        D_fake = G_I_LR.detach()  # Detach to prevent backpropigation through G
        fake_labels = torch.zeros_like(D_fake)
        loss_D_fake = nn.BCELoss()(D_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake

        return loss_D

    # def compute_G_loss(self, D, G, I_LR) -> float:
    #     """
    #     Compute adversarial loss for Generator.
    #     Real image targets == 1, fake images == 0.

    #     Args:
    #         D: Discriminator Network
    #         G: Generator Network
    #         I_LR: Batch of low-resolution images (tensors)

    #     Returns:
    #         loss_G: Generator's adversarial loss
    #     """
    #     G_I_LR = G(I_LR)
    #     D_fake = D(G_I_LR)
    #     real_labels = torch.ones_like(D_fake)
    #     loss_G = nn.BCELoss()(D_fake, real_labels)

    #     return loss_G

    # FIXME. confer w/ paper to correct this.
    def forward(self) -> torch.Tensor:
        return self.loss_d


class PerceptualLoss(nn.Module):

    def __init__(self, D_real, G_I_LR):
        """Weighted sum of Content Loss & Adversarial Loss."""
        super(PerceptualLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss(D_real, G_I_LR)

    def forward(self):
        return torch.Tensor(self.content_loss) + torch.mul(
            torch.Tensor(self.adversarial_loss), (10**-3)
        )  # FIXME

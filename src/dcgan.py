"""Implimentation of Deep Convolutional Generative Adversarial Network."""

from torch import nn


class G_block():
    """Generator Block."""
    def __init__(self,  out_channels, in_channels=3, kernel_size=4, stride=2, padding=1, **kwargs):
        super(G_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))
    
class D_block():
    """Discriminator Block."""
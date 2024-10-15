"""Pytorch implimentation of SRGAN model, described here:
https://arxiv.org/pdf/1609.04802
"""

from torch import nn


class SRGAN(nn.Module):
    """Super Resolution Generative Adversarial Network."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

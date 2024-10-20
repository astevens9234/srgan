"""Pytorch implimentation of SRGAN model, described here:
c.f. <https://arxiv.org/pdf/1609.04802>
"""

from torch import nn


class SRGAN(nn.Module):
    """Super Resolution Generative Adversarial Network."""

    def __init__(self):
        super(SRGAN, self).__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError


class G_Block(nn.Module): ...


class D_Block(nn.Module): ...


class ContentLoss(nn.Module):
    """19 Layer VGG loss purposed at: <https://arxiv.org/pdf/1409.1556>"""

    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self):
        raise NotImplementedError


class AdversarialLoss(nn.Module):
    """"""

    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self):
        raise NotImplementedError


class PerceptualLoss(nn.Module):
    """Weighted sum of Content Loss & Adversarial Loss."""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.adversarial_loss = AdversarialLoss()

    def forward(self):
        return self.content_loss + ((10**-3) * self.adversarial_loss)

"""Implimenation of a simple GAN. Educational - to inform the larger problem.

Summary:
- Generative Adversarial Networks are composed of two neural networks;
    the Generator and Discriminator.
- The Generator generates a sample as close as possible to a true sample
    to fool the Discriminator, by maxing cross-entropy loss.
- The Discriminator tries to distinguish between the true sample from
    the generated sample, by minimizing the cross-entropy loss.
"""

import seaborn as sns
import torch

from matplotlib import pyplot as plt
from torch import nn


# Loading some stock data
A = torch.tensor([[1, 2], [-0.1, 0.5]])
X = torch.normal(0.0, 1, (1000, 2))
b = torch.tensor([1, 2])

data = torch.matmul(X, A) + b

# Visual for reference
if False:
    data_scatterplot = sns.scatterplot(data[:100, 0])
    fig = data_scatterplot.get_figure()
    fig.savefig("out.png")

# single layer linear Generator
net_G = nn.Sequential(nn.Linear(2, 2))

# three layer MLP Discriminator
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 1)
)


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Training
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update the Disciminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)

    trainer_D.zero_grad()

    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())

    loss_D = (
        loss(real_Y, ones.reshape(real_Y.shape))
        + loss(fake_Y, zeros.reshape(fake_Y.shape))
    ) / 2

    loss_D.backward()
    trainer_D.step()
    return loss_D


def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update the Generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)

    trainer_G.zero_grad()

    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)

    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


def training(net_D, net_G, lr_D, lr_G, epochs, data_iter, latent_dim):
    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_D.parameters(), lr=lr_G)

    for _ in range(epochs):
        metric = Accumulator(3)
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(
                update_D(X, Z, net_D, net_G, loss, trainer_D),
                update_G(Z, net_D, net_G, loss, trainer_G),
                batch_size,
            )
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        print("loss_D: {}, loss_G {}".format(loss_D, loss_G))


lr_D, lr_G, latent_dim, epochs = 0.05, 0.005, 2, 20
batch_size = 8
data_iter = torch.utils.data.DataLoader((data,), batch_size)

training(
    net_D, net_G, lr_D, lr_G, epochs, data_iter, latent_dim
)

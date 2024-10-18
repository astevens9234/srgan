"""Common classes/functions/etc for our GANs."""

import torch
from torch import nn

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
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
"""Header level training script for SRGAN.

To execute, spin up a virtual environment and run the command:
py ./src/srgan_trianing.py --config config.yaml
^ add ability to pass config. c.f. argparse library
"""

import datetime as dt
import logging
import yaml

import torch

from torch import nn
from srgan import SRGAN, PerceptualLoss


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
run_name = config["training"]["run_name"]

net = SRGAN(resize=(64, 64))


def training_loop(generator, discriminator, device, epochs, learning_rate):
    """
    Args:
        generator: generator network
        discriminator: discriminator network
        device: chip to run training on
        epochs: number of training iterations
        learning_rate: optimizer parameter
    """

    for w in generator.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in discriminator.parameters():
        nn.init.normal_(w, 0, 0.02)

    generator, discriminator = generator.to(device), discriminator.to(device)
    grid = {"lr": learning_rate}
    generator_trainer = torch.optim.Adam(generator.parameters(), **grid)
    discriminator_trainer = torch.optim.Adam(discriminator.parameters(), **grid)

    # FIXME: What is the succinct way to pass batches back and forth?
    loss = PerceptualLoss(generator, discriminator)

    for _ in range(0, epochs):
        ...

    ts = dt.datetime.now()
    torch.save(
        generator.state_dict(),
        "./models/srgan-generator-{}-{}.params".format(run_name, ts),
    )
    torch.save(
        discriminator.state_dict(),
        "./models/srgan-discriminator-{}-{}.params".format(run_name, ts),
    )

    # TODO: Quality Logging & perhaps Profiling


def main():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    training_loop(net.generator, net.discriminator, device, epochs, learning_rate)

    # TODO: Generate summary of training run...


if __name__ == "main":
    main()

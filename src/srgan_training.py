"""Header level training script for SRGAN.

To execute, spin up a virtual environment and run the command:
py ./src/srgan_trianing.py --config config.yaml
^ add ability to pass config. c.f. argparse library
"""

# NOTE: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

import datetime as dt
import logging
import yaml

import torch

from torch import nn
from torchvision import models, transforms, datasets

from srgan import SRGAN, PerceptualLoss


with open("./src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
run_name = config["training"]["run_name"]


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

    # FIXME: Now I need to correct for I_HR & I_LR coming from the DataLoader.
    # D_real = discriminator(I_HR)
    # G_I_LR = generator(I_LR)

    # loss = PerceptualLoss(D_real, G_I_LR)

    for _ in range(0, epochs):
        ...

#     
# 
# 
#     

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

    data = datasets.ImageNet("../data", transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True) #, num_workers=4)

    print("hit")

    for X, y in data_loader:
        print("Shape of X [N, C, H, W]: ".format(X.shape))
        print(y.shape)

    net = SRGAN()

    training_loop(net.generator, net.discriminator, device, epochs, learning_rate)

    # TODO: Generate summary of training run...


if __name__ == "__main__":
    main()

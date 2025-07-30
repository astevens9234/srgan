"""Header level training script for SRGAN.

To execute, spin up a virtual environment and run the command:
py ./src/srgan_trianing.py --config config.yaml
^ add ability to pass config. c.f. argparse library
"""

# NOTE: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

import datetime as dt
import logging
import warnings
import yaml

import torch

from torch import nn
from torchvision import models, transforms, datasets
from torchsummary import summary

from srgan import SRGAN, PerceptualLoss

warnings.simplefilter(action="ignore")

with open("./src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# NOTE: https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
betas = config["training"]["betas"]
dataset = config["data"]["dataset"]
epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
run_name = config["training"]["run_name"]


def training_loop(generator, discriminator, dataloader, device, epochs, learning_rate, batch_size, betas):
    """
    Args:
        generator: generator network
        discriminator: discriminator network
        device: chip to run training on
        epochs: number of training iterations
        learning_rate: rate of change updating model parameters per batch
        batch_size: number of samples per batch
    """

    for w in generator.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in discriminator.parameters():
        nn.init.normal_(w, 0, 0.02)

    grid = {"lr": learning_rate, "betas":betas}
    generator_trainer = torch.optim.Adam(generator.parameters(), **grid)
    discriminator_trainer = torch.optim.Adam(discriminator.parameters(), **grid)

    """ Leaving off here 07/29
    - Both networks are operational.
    - Need to flesh out the training loop.
        - prediction w/ generator
        - classification w/ discriminator
        - think I need to rework the loss functions.
    """

    lossfx = PerceptualLoss()

    # Downsampling I_HR
    # NOTE: This reduces cost of training, at the expense of performance.
    pool = nn.AvgPool2d(kernel_size=4, stride=4)

    generator.train()
    discriminator.train()

    for batch, (X, _) in enumerate(dataloader):

        print(batch)

        X = X.to(device)
        I_LR = pool(X)
        G_I_LR = generator(I_LR)
        D_real = discriminator(G_I_LR)

        # loss = lossfx(D_real, G_I_LR)

        # Backprop
        # generator_trainer.zero_grad()
        # discriminator_trainer.zero_grad()
        # loss.backward()
        # generator_trainer.step()
        # discriminator_trainer.step()

        # print(f"loss: {loss}")

    #
    #
    #
    #

    # ts = dt.datetime.now()
    # torch.save(
    #     generator.state_dict(),
    #     "./models/srgan-generator-{}-{}.params".format(run_name, ts),
    # )
    # torch.save(
    #     discriminator.state_dict(),
    #     "./models/srgan-discriminator-{}-{}.params".format(run_name, ts),
    # )

    # TODO: Quality Logging & perhaps Profiling

    print("close")


def main():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = datasets.ImageFolder(
        root=f"~/data/{dataset}/train", transform=transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    generator = SRGAN().generator.to(device)
    discriminator = SRGAN().discriminator.to(device)

    if False:  # Sanity Checks
        for X, y in data_loader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            
        summary(generator, input_size=(3, 64, 64))
        summary(discriminator, input_size=(3, 64, 64))

    training_loop(generator, discriminator, data_loader, device, epochs, learning_rate, batch_size, betas)


if __name__ == "__main__":
    main()

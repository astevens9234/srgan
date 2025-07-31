"""Header level training script for SRGAN."""

# NOTE: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

import datetime as dt
import logging
import os
import warnings
import yaml

import torch

from torch import nn
from torchvision import transforms, datasets
from torchsummary import summary

from srgan import SRGAN, PerceptualLoss

with open("./src/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# NOTE: https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
betas = config["training"]["betas"]
dataset = config["data"]["dataset"]
epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
batch_size = config["training"]["batch_size"]
run_name = config["training"]["run_name"]
note = config["note"]

warnings.simplefilter(action="ignore")
logging.basicConfig(filename=f"./logs/srgan-{run_name}.log", encoding="utf-8", level=logging.INFO)

def training_loop(
    generator,
    discriminator,
    dataloader,
    device,
    learning_rate,
    batch_size,
    betas,
    epochs
):
    """
    Args:
        generator: generator network
        discriminator: discriminator network
        dataloader: pytorch DataLoader object
        device: chip to run training on
        learning_rate: rate of change updating model parameters per batch
        batch_size: number of samples per batch
        betas: coefficients for optimizer
        epochs: number of runs through the dataloader
    """

    for w in generator.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in discriminator.parameters():
        nn.init.normal_(w, 0, 0.02)

    grid = {"lr": learning_rate, "betas": betas}
    generator_trainer = torch.optim.Adam(generator.parameters(), **grid)
    discriminator_trainer = torch.optim.Adam(discriminator.parameters(), **grid)

    lossfx = PerceptualLoss()
    size = len(dataloader.dataset)

    # Downsampling I_HR
    # NOTE: This reduces cost of training, at the expense of performance.
    pool = nn.AvgPool2d(kernel_size=4, stride=4)

    generator.train()
    discriminator.train()

    for e in range(0, epochs):
        logging.info(f"Epoch {e+1}\n{"*"*20}")

        for batch, (X, _) in enumerate(dataloader):

            X = X.to(device)

            if X.shape != torch.Size([64, 3, 64, 64]):
                continue

            with torch.autocast(device_type=device):
                I_LR = pool(X)
                I_SR = generator(I_LR)
                D_pred = discriminator(I_SR)

                loss = lossfx(D_pred=D_pred, I_SR=I_SR, X=X)

            generator_trainer.zero_grad()
            discriminator_trainer.zero_grad()
            loss.backward()
            generator_trainer.step()
            discriminator_trainer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                # Checkpoints
                torch.save(
                    generator.state_dict(),
                    f"./srgan-generator-{run_name}.params",
                )
                torch.save(
                    discriminator.state_dict(),
                    f"./srgan-discriminator-{run_name}.params",
                )




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

    logging.info(f"{run_name}\n{note}\n")

    training_loop(
        generator,
        discriminator,
        data_loader,
        device,
        learning_rate,
        batch_size,
        betas,
        epochs
    )


if __name__ == "__main__":
    main()

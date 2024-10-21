"""Image Upscaling Research, developed in Pytorch.

Right now, this file is debt from earlier builds - moving everything out eventually.
"""

import os
import zipfile

import requests
import torch
import torchvision

from src.dcgan import G_block


def upscale(device, img="nature.Jpeg", model="./models/dcgan-netg.params"):
    """This function will read an image and save it in super resolution..."""
    # TODO:
    #       call generator/other F(x)s for SR
    #       upscale
    #       save file...
    X = torchvision.io.read_image(img).type(torch.FloatTensor).to(device)

    # NOTE: Doesn't exactly make sense to make predictions off of the DCGAN model
    #       because it is trained on toy data.
    # upscaler = G_block(out_channels=3).to(device)
    # upscaler.load_state_dict(torch.load(model, weights_only=True), strict=False)
    # print(f"X type {type(X)}")
    # Y = upscaler(X)
    # res = torchvision.transforms.ToPILImage()(Y)
    # res.show()

    raise NotImplementedError


def main():
    """Image Upscaler based on DCGAN/SRGAN/ETC"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    upscale(device=device)


if __name__ == "__main__":
    main()

"""Super Resolution Generative Adversarial Network.

Upscaling images.
"""

import torch

from src.srgan import SRGAN




def training():
    raise NotImplementedError

def testing():
    raise NotImplementedError

def main():
    """"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
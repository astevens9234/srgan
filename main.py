"""I am the script, you are the user.
"""

import os
import zipfile

import requests
import torch

from src.srgan import SRGAN


def _download_zip(url, folder='./data') -> str:
    print('Downloading...')

    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, url.split('/')[-1])
    r = requests.get(url, stream=True, verify=True)
    print(f"Status code {r.status_code}")

    with open(filename, 'wb') as f:
        f.write(r.content)

    return filename

def extract_zip(url, folder=None):
    filename = _download_zip(url=url, folder=folder)
    print('Extracting...')

    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)

    assert ext in ('.zip'), 'Only support .zip file at this time'

    filepath = zipfile.ZipFile(filename, 'r')
    if folder is None:
        folder = base_dir

    filepath.extractall(folder)


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

    # Some stock data...
    extract_zip(url='http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip', folder='./data')

if __name__ == '__main__':
    main()

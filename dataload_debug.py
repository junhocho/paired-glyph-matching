import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb, TwoGlyphsPerFont

import torchvision.transforms as transforms


numbers = "0123456789"
lowercases = "abcdefghijklmnopqrstuvwxyz"
uppercases = lowercases.upper()

image_dir = 'data/explor_all/image'
attribute_path = 'data/explor_all/attributes.txt'
dataset_name = 'explor_all'
img_size = 64
n_style = 4



test_dataloader = get_loader(image_dir, attribute_path,
                             dataset_name=dataset_name,
                             image_size=img_size,
                             batch_size=52,
                             mode='test', binary=False)

# batch = next(iter(test_dataloader))
# style_ref = batch['styles_A']
# img1 = batch['styles_A'][0][:3,...]
# img2 = batch['styles_A'][0][3:6,...]
# img3 = batch['styles_A'][0][6:9,...]
# img4 = batch['styles_A'][0][9:,...]

import pdb;pdb.set_trace()
# batch = next(iter(test_dataloader))

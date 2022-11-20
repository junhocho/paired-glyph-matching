import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb, TwoGlyphsPerFont

import torchvision.transforms as transforms
from contextlib import redirect_stdout
from io import StringIO


numbers = "0123456789"
lowercases = "abcdefghijklmnopqrstuvwxyz"
uppercases = lowercases.upper()

# image_dir = 'data/explor_all/image'
# attribute_path = 'data/explor_all/attributes.txt'
# dataset_name = 'donovan_embedding'
# dataset_name = 'donovan_embedding_per_char'

image_dir = './data/ofl_images/'
attribute_path = "./data/ofl_images/glyph_files.txt"
dataset_name = 'ofl_per_char'

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="model folder or pth path")
    parser.add_argument("--test_num", type=int, default=20, help="num of testset")
    # Other Modules
    return parser
parser = get_parser()
opts = parser.parse_args()
# exp_path = "experiments/AUG-simclr-h70/checkpoint/F_8500.pth"
# exp_path = "experiments/font-cls/checkpoint/F_900.pth"
# exp_path = "experiments/font-cls-attr/checkpoint/F_1200.pth"
# exp_path = "experiments/AUG-simclr-h70-Attr/checkpoint/F_7400.pth"
exp_path = opts.model_path
test_num = opts.test_num


img_size = 64
n_style = 4


test_dataloader = get_loader_fontemb(image_dir, attribute_path,
                             dataset_name=dataset_name,
                             image_size=img_size,
                             batch_size=52,
                             mode='test', binary=False)

batch_test = next(iter(test_dataloader))

import network
from network.networks import ResNet34, ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fontemb_net = ResNet18(3, 512, None, [70], use_simclr_head=False)  ## PUI
fontemb_net = ResNet18(3, 512, None, [70], use_simclr_head=True)  ## simclr
fontemb_net = fontemb_net.to(device)

with redirect_stdout(StringIO()) as f:
    fontemb_net.load_state_dict(torch.load(exp_path))

fontemb_net.eval()

features = []
import time
t = time.time()

with torch.no_grad():
    for ii, batch_test in enumerate(test_dataloader):
        img_i = batch_test['img_i'].to(device)
        feat_i, _  = fontemb_net(img_i)
        features.append(feat_i)
print((time.time() - t)/60, 'min')


features_per_font = torch.stack(features) ##  torch.Size([100, 52, 512])
feat_per_char = features_per_font.permute(1,0,2) #  torch.Size([52, 100, 512])



from evaluation import retrieval_evaluation
ret_accuracy, ret_per_char = retrieval_evaluation(feat_per_char[:, :test_num, :], test_dataloader, init_char=0)
print(ret_accuracy)

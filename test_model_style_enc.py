import sys
import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb, TwoGlyphsPerFont

import torchvision.transforms as transforms

from evaluation import retrieval_evaluation



from options import get_parser

parser = get_parser()
opts = parser.parse_args()

print(opts)

numbers = "0123456789"
lowercases = "abcdefghijklmnopqrstuvwxyz"
uppercases = lowercases.upper()


img_size = 64
n_style = 4


if opts.dataset_name=='donovan':
    image_dir = 'data/explor_all/image'
    attribute_path = 'data/explor_all/attributes_alphanumeric.txt'
    dataset_name = 'donovan_embedding_per_char'
    test_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=52,
                                 mode='test', binary=False)
elif opts.dataset_name=='ofl':
    image_dir = './data/ofl_images/'
    attribute_path = "./data/ofl_images/glyph_files.txt"
    dataset_name = 'ofl_per_char'
    test_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=52,
                                 mode='test', binary=False, test_num=opts.test_num)




import network
from network.networks import ResNet34, ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from embed_with_styleenc import load_style_enc
style_enc = load_style_enc(opts)
style_enc = style_enc.to(device)
style_enc.eval()


char_num = test_dataloader.dataset.char_num
feat_per_char = []

def evaluate_ret_acc(style_enc):
    # load
    
    # infer
    import time
    t = time.time()

    c_list = []
    char_num = test_dataloader.dataset.char_num

    feat_per_char = []
    ret_dict = {}

    features = []
    with torch.no_grad():
        for ii, batch_test in enumerate(test_dataloader):
            img_i = batch_test['img_i'].to(device)
            img_i = img_i.repeat(1,opts.n_style,1,1)
            feat_i = style_enc.infer_style(img_i).squeeze(2).squeeze(2)
            features.append(feat_i)

    features_per_font = torch.stack(features) ##  torch.Size([100, 52, 512])
    feat_per_char = features_per_font.permute(1,0,2) #  torch.Size([52, 100, 512])

    ret_accuracy_upper, ret_per_char_upper = retrieval_evaluation(feat_per_char, test_dataloader, init_char=26)
    ret_accuracy, ret_per_char = retrieval_evaluation(feat_per_char, test_dataloader, init_char=0)

    return ret_accuracy, ret_per_char, ret_accuracy_upper, ret_per_char_upper




ret_accuracy_per_epoch = {}

t = time.time()


rets = evaluate_ret_acc(style_enc)
ret_accuracy, ret_per_char, ret_accuracy_upper, ret_per_char_upper = rets 
print("allcases:", ret_accuracy)
print("uppercases:", ret_accuracy_upper)

ret_accuracy_per_epoch = rets
    
print((time.time() - t)/60, "min")

# import json
# json_file = opts.model_path
# if opts.memo:
#     json_file += '-'
#     json_file += opts.memo
# json_file = '{}.json'.format(json_file)
# with open(json_file, 'w') as fp:
#     json.dump(ret_accuracy_per_epoch, fp)

# xx = json.load(open('AUG-simclr-h70-refactored2.json'))
# max(xx, key=lambda key: xx[key][0])

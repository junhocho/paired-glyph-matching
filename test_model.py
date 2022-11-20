import sys
import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb, TwoGlyphsPerFont

import torchvision.transforms as transforms

from evaluation import evaluate_ret_acc

from options import get_parser
parser = get_parser()
opts = parser.parse_args()

print(opts)
model_path = opts.pretrained

numbers = "0123456789"
lowercases = "abcdefghijklmnopqrstuvwxyz"
uppercases = lowercases.upper()

image_dir = 'data/explor_all/image'
attribute_path = 'data/explor_all/attributes_alphanumeric.txt'

# dataset_name = 'donovan_embedding'
dataset_name = 'donovan_embedding_per_char'

img_size = 64
test_num=100

if opts.dataset_name == 'donovan':
    char_set = 'alphabets'
    batch_size = 52
    val_dataloader = get_loader_fontemb(
                                'data/explor_all/image',
                                'data/explor_all/attributes_alphanumeric.txt',
                                 dataset_name='donovan_embedding_per_char',
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='test', binary=False, char_set=char_set)
    test_dataloader = get_loader_fontemb(
                                './data/ofl_images/',
                                "./data/ofl_images/glyph_files.txt",
                                dataset_name='ofl_per_char',
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='val', binary=False, char_set=char_set, test_num=test_num)
    test2_dataloader = None
elif opts.dataset_name == 'OFL':
    char_set = 'alphabets'
    batch_size = 52
    val_dataloader = get_loader_fontemb(
                                './data/ofl_images/',
                                "./data/ofl_images/glyph_files.txt",
                                dataset_name='ofl_per_char',
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='val', binary=False, char_set=char_set, test_num=test_num)
    test_dataloader = get_loader_fontemb(
                                'data/explor_all/image',
                                'data/explor_all/attributes_alphanumeric.txt',
                                 dataset_name='donovan_embedding_per_char',
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='test', binary=False, char_set=char_set)
    test2_dataloader = None
elif opts.dataset_name == 'Capitals64':
    char_set = 'capitals'
    batch_size = 26
    # val_dataloader = get_loader_fontemb(
    #                             'data/explor_all/image',
    #                             'data/explor_all/attributes_alphanumeric.txt',
    #                              dataset_name='donovan_embedding_per_char',
    #                              image_size=img_size,
    #                              batch_size=batch_size,
    #                              mode='test', binary=False, char_set=char_set)
    val_dataloader = get_loader_fontemb(
                                "./data/Capitals64_split/",
                                None,
                                dataset_name = "Capitals64_per_char",
                                image_size=img_size,
                                batch_size=batch_size,  
                                char_set=char_set,
                                mode='val', binary=False)
    test_dataloader = get_loader_fontemb(
                                "./data/Capitals64_split/",
                                None,
                                dataset_name = "Capitals64_per_char",
                                image_size=img_size,
                                batch_size=batch_size,  
                                char_set=char_set,
                                mode='test', binary=False)
    test2_dataloader = get_loader_fontemb(
                                'data/explor_all/image',
                                'data/explor_all/attributes_alphanumeric.txt',
                                 dataset_name='donovan_embedding_per_char',
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='train_test', binary=False, char_set=char_set)
else:
    raise NotImplementedError('Unknown dataset : {}'.format(opts.dataset_namae)) 

import network
from network.networks import ResNet34, ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fontemb_net = ResNet18(3, 512, None, [70], use_simclr_head=False)  ## PUI
# fontemb_net = ResNet18(3, opts.feat_dim, None, [70], use_simclr_head=True)  ## simclr

if opts.backbone == 'ResNet34':
    fontemb_net = ResNet34(3, opts.feat_dim, None, opts.heads,
            norm_method=opts.norm_method,
            use_simclr_head=opts.simclr or opts.supcon
            )  ## input is black white but 3 channel
elif opts.backbone == 'ResNet18':
    fontemb_net = ResNet18(3, opts.feat_dim, None, opts.heads,
            norm_method=opts.norm_method,
            use_simclr_head=opts.simclr or opts.supcon
            )  ## input is black white but 3 channel


fontemb_net = fontemb_net.to(device)
fontemb_net.eval()

# model_path = 'AUG-simclr-h70'
# model_path = "F_4900_ofl_pret.pth"

if model_path.endswith("pth"):
    chkpt_path = model_path
    f = chkpt_path.split('/')[-1]
    epochs = [int(f.split('_')[1].split('.pth')[0])]
    F_files = [chkpt_path]

else:
    # exp_path = "./AUG-simclr-h70-debug"
    exp_path = model_path


    chkpt_path = os.path.join(exp_path, 'checkpoint')
    chkpt_files = os.listdir(chkpt_path)
    epochs = sorted([int(f.split('_')[1].split('.pth')[0]) for f in chkpt_files if f.startswith("F")])
    F_files = [os.path.join(chkpt_path,"F_{}.pth".format(e)) for e in epochs]


feat_per_char = []



ret_accuracy_per_epoch = {}

t = time.time()

if len(epochs) > 1:
    import wandb
    use_wandb = True
    wandb.init(project="donovan_embedding")
    wandb.run.name = opts.pretrained.split('experiments/')[-1].split('/')[0]
    wandb.config.update(opts)
else:
    use_wandb = False


def evaluate(dataloader):
    best_ret_accuracy = 0 
    for  ii, (e, F) in enumerate(zip(epochs, F_files)):
        print("== {}/{} epoch: {} ==".format(e, epochs[-1], F))
        dataset_name = dataloader.dataset.dataset_name
        print("{} set".format(dataset_name))
        rets = evaluate_ret_acc(fontemb_net, F, dataloader)
        # ret_accuracy, ret_per_char, ret_accuracy_upper, ret_per_char_upper = rets 
        ret_accuracy, ret_per_char = rets 

        print("allcases:", ret_accuracy)
        # print("uppercases:", ret_accuracy_upper)

        if ret_accuracy > best_ret_accuracy:
            best_ret_accuracy = ret_accuracy 
            if use_wandb:
                wandb.log({
                    "best retrieval accuracy" : best_ret_accuracy,
                    "best epoch" : e,
                    })
        if use_wandb:
            wandb.log({
                'epoch' : e,
                'retrieval accuracy_all {}'.format(dataset_name) : ret_accuracy
                })

        ret_accuracy_per_epoch[e] = rets
        
    print((time.time() - t)/60, "min for {} tests".format(ii))


if test2_dataloader:
    evaluate(test2_dataloader)
evaluate(val_dataloader)
evaluate(test_dataloader)

# ## save json
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

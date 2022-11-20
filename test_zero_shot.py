import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb, TwoGlyphsPerFont

from contextlib import redirect_stdout
from io import StringIO


numbers = "0123456789"
lowercases = "abcdefghijklmnopqrstuvwxyz"
uppercases = lowercases.upper()
alphabets = lowercases+uppercases

img_size = 64
n_style = 4

from options import get_parser
parser = get_parser()
opts = parser.parse_args()


if opts.dataset_name == 'donovan':
    image_dir = 'data/explor_all/image'
    attribute_path = 'data/explor_all/attributes_alphanumeric.txt'
    # dataset_name = 'donovan_embedding'
    dataset_name = 'donovan_embedding_per_char'

    char_set = 'numbers'
    batch_size = 10

    test_numbers_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='test', binary=False, char_set=char_set)

    char_set = 'alphabets'
    batch_size = 52

    test_alphabets_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='test', binary=False, char_set=char_set)
elif opts.dataset_name == 'ofl':
    image_dir = './data/ofl_images/'
    attribute_path = "./data/ofl_images/glyph_files.txt"
    dataset_name = 'ofl_per_char'


    char_set = 'numbers'
    batch_size = 10

    test_numbers_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='val', binary=False, char_set=char_set, test_num=100)

    char_set = 'alphabets'
    batch_size = 52

    test_alphabets_dataloader = get_loader_fontemb(image_dir, attribute_path,
                                 dataset_name=dataset_name,
                                 image_size=img_size,
                                 batch_size=batch_size,
                                 mode='val', binary=False, char_set=char_set, test_num=100)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(opts):
    # if opts.simclr or opts.train_fontcls:
    if opts.phase == 'test-representation':
        import network
        from network.networks import ResNet34, ResNet18

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

        with redirect_stdout(StringIO()) as f:
            fontemb_net.load_state_dict(torch.load(opts.pretrained))

        fontemb_net.eval()
        model = fontemb_net

    # else:
    #     from embed_with_styleenc import load_style_enc
    #     styleenc_net = load_style_enc(opts)
    #     styleenc_net = styleenc_net.to(device)
    #     styleenc_net.eval()
    #     model = styleenc_net

    return model


def evaluate_model(opts, model):
    features = []
    import time
    t = time.time()

    with torch.no_grad():
        for ii, batch_test in enumerate(test_alphabets_dataloader):
            img_i = batch_test['img_i'].to(device)
            if opts.phase == 'test-representation':
                feat_i, _  = model(img_i)
            else:
                img_i = img_i.repeat(1,opts.n_style,1,1)
                feat_i = model.infer_style(img_i).squeeze(2).squeeze(2)

            features.append(feat_i)
    print((time.time() - t)/60, 'min')
    features_per_font = torch.stack(features)
    print(features_per_font.shape)
    feat_per_char = features_per_font.permute(1,0,2) # torch.Size([52, 28, 512])

    features = []
    import time
    t = time.time()

    with torch.no_grad():
        for ii, batch_test in enumerate(test_numbers_dataloader):
            img_i = batch_test['img_i'].to(device)
            if opts.phase == 'test-representation':
                feat_i, _  = model(img_i)
            else:
                img_i = img_i.repeat(1,opts.n_style,1,1)
                feat_i = model.infer_style(img_i).squeeze(2).squeeze(2)
            features.append(feat_i)
    print((time.time() - t)/60, 'min')
    features_per_font = torch.stack(features) 
    print(features_per_font.shape)
    feat_per_num = features_per_font.permute(1,0,2) #  torch.Size([10, 28, 512])

    num_font = feat_per_num.size(1)
    c_list = []
    ret_dict = {}
    with torch.no_grad():
        for ii in range(feat_per_num.size(0)):
            for jj in range(feat_per_char.size(0)):
                feat_num = feat_per_num[ii]
                feat_char = feat_per_char[jj]
                cdist = torch.cdist(feat_num, feat_char, 2)
                
                ## num --> char
                retrieval_prediction = list(torch.argmin(cdist, dim=1).cpu().numpy())
                c = 0
                for i in range(num_font):
                    if i == retrieval_prediction[i]:
                        c += 1
                # print("{}/{} correct".format(c, num_font))
                c_list.append(c)
                ret_dict["{}->{}".format(numbers[ii], alphabets[jj])] = c

                ## char --> num
                retrieval_prediction = list(torch.argmin(cdist, dim=0).cpu().numpy())
                c = 0
                for i in range(num_font):
                    if i == retrieval_prediction[i]:
                        c += 1
                # print("{}/{} correct".format(c, num_font))
                c_list.append(c)
                ret_dict["{}->{}".format(alphabets[jj], numbers[ii])] = c

    ret_accuracy = sum(c_list)/len(c_list)/num_font*100
    print(ret_accuracy)

    ret_per_char = {}
    for ii in range(len(alphabets)):
        c = alphabets[ii]
        c_ret = [v for k, v in ret_dict.items() if k.startswith(c)]
        c_ret = sum(c_ret)/len(c_ret)/num_font*100
        ret_per_char[c] = c_ret
    ret_per_char = {e[0]+"_ret_accuracy":e[1] for e in sorted(ret_per_char.items(), key=lambda x: x[1])}
    ret_per_char

    ret_per_num = {}
    for ii in range(len(numbers)):
        c = numbers[ii]
        c_ret = [v for k, v in ret_dict.items() if k.startswith(c)]
        c_ret = sum(c_ret)/len(c_ret)/num_font*100
        ret_per_num[c] = c_ret
    ret_per_num = {e[0]+"_ret_accuracy":e[1] for e in sorted(ret_per_num.items(), key=lambda x: x[1])}
    ret_per_num


    x = ([v for k, v in ret_per_num.items()])
    num2char = sum(x)/len(x)
    print("(0~9) → (a~Z):", num2char)
    x = ([v for k, v in ret_per_char.items()])
    char2num = sum(x)/len(x)
    print("(a~Z) → (0~9):", char2num)
    return num2char, char2num


if opts.pretrained.endswith("pth"):
    model = load_model(opts)
    num2char, char2num = evaluate_model(opts, model)
else:
    exp_path = opts.pretrained

    chkpt_path = os.path.join(exp_path, 'checkpoint')
    chkpt_files = os.listdir(chkpt_path)
    epochs = sorted([int(f.split('_')[1].split('.pth')[0]) for f in chkpt_files if f.startswith("F")])
    F_files = [os.path.join(chkpt_path,"F_{}.pth".format(e)) for e in epochs]

    import wandb
    wandb.init(project="donovan_embedding")
    wandb.run.name = opts.pretrained.split('experiments/')[-1].split('/')[0]
    wandb.config.update(opts)
    best_num2char = 0
    best_char2num = 0
    for  ii, (e, F) in enumerate(zip(epochs, F_files)):
        opts.pretrained = F
        print('epoch : {} , load {}'.format(e, F))
        model = load_model(opts)
        num2char, char2num = evaluate_model(opts, model)
        wandb.log({
            'epoch' : e,
            'num2char_{}'.format(test_alphabets_dataloader.dataset.dataset_name) : num2char,
            'char2num_{}'.format(test_alphabets_dataloader.dataset.dataset_name) : char2num,
            })
        if num2char > best_num2char:
            best_num2char = num2char
            best_num2char_epoch = e

        if char2num > best_char2num:
            best_char2num = char2num
            best_char2num_epoch = e
    wandb.log({
        'best_num2char_epoch' : best_num2char_epoch,
        'best_num2char_{}'.format(test_alphabets_dataloader.dataset.dataset_name) : best_num2char,
        })

    wandb.log({
        'best_char2num_epoch' : best_char2num_epoch,
        'best_char2num_{}'.format(test_alphabets_dataloader.dataset.dataset_name) : best_char2num,
        })

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

char_set = 'alphabets'
from options import get_parser
parser = get_parser()
opts = parser.parse_args() 

if opts.dataset_name == 'ofl':
    ofl_val_dataloader = get_loader_fontemb(
                                './data/ofl_images/',
                                "./data/ofl_images/glyph_files.txt",
                                dataset_name='ofl_per_char',
                                 image_size=64,
                                 batch_size=52,
                                 mode='val', binary=False, char_set=char_set, test_num=100)
    val_dataloader = ofl_val_dataloader
elif opts.dataset_name  == 'donovan':
    donovan_val_dataloader = get_loader_fontemb(
                            'data/explor_all/image',
                            'data/explor_all/attributes_alphanumeric.txt',
                             dataset_name='donovan_embedding_per_char',
                             image_size=64,
                             batch_size=52,
                             mode='test', binary=False, char_set=char_set)
    val_dataloader = donovan_val_dataloader







from network.networks import ResNet34, ResNet18, MLP
fontemb_net = ResNet18(3, opts.feat_dim, None, opts.heads,
        norm_method=opts.norm_method,
        use_simclr_head=opts.simclr or opts.supcon
        )  ## input is black white but 3 channel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fontemb_net = fontemb_net.to(device)

num_chars = val_dataloader.dataset.char_num
idx2chars = val_dataloader.dataset.idx2chars

from model import FontDecoder
fontdec_net = FontDecoder(in_channel=opts.feat_dim + num_chars, out_channel=3, attention=True).to(device)

# fontemb_net.load_state_dict(torch.load('./F_27300.pth'))
# fontdec_net.load_state_dict(torch.load('./Fontdec_27300.pth'))

fontemb_net.eval()
fontdec_net.eval()

# L1_error_per_char_all = []



from contextlib import redirect_stdout
from io import StringIO
def load_model(opts, epoch):
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

        fontdec_net = FontDecoder(in_channel=opts.feat_dim + num_chars, out_channel=3, attention=True).to(device)

        with redirect_stdout(StringIO()) as f:
            fontemb_net.load_state_dict(torch.load(
                os.path.join(opts.pretrained, 'checkpoint', 'F_{}.pth'.format(epoch))))
            fontdec_net.load_state_dict(torch.load(
                os.path.join(opts.pretrained, 'checkpoint', 'Fontdec_{}.pth'.format(epoch))))

        fontemb_net.eval()
        fontdec_net.eval()


    return fontemb_net, fontdec_net

model_path = opts.pretrained
if model_path.endswith("pth"):
    chkpt_path = model_path
    f = chkpt_path.split('/')[-1]
    epochs = [int(f.split('_')[1].split('.pth')[0])]
    opts.pretrained = opts.pretrained.split('/checkpoint')[0]
else:
    exp_path = model_path
    chkpt_path = os.path.join(exp_path, 'checkpoint')
    chkpt_files = os.listdir(chkpt_path)
    epochs = sorted([int(f.split('_')[1].split('.pth')[0]) for f in chkpt_files if f.startswith("F_")])
print(epochs)

use_wandb=False
if len(epochs) > 1:
    use_wandb=True
    import wandb
    wandb.init(project="donovan_embedding")
    wandb.run.name = opts.pretrained.split('experiments/')[-1].split('/')[0]
    wandb.config.update(opts)

from evaluation import L1_gen_evaluation
for epoch in epochs:
    if epoch < opts.init_epoch:
        continue
    if not epoch % opts.log_freq == 0:
        continue

    print('epoch : {} '.format(epoch))

    fontemb_net, fontdec_net = load_model(opts, epoch)

    L1_error = L1_gen_evaluation(fontemb_net, fontdec_net, val_dataloader, True)

    print("mean epoch {} error {}".format(epoch, L1_error))

    if use_wandb:
        wandb.log({
            'epoch' : epoch,
            'L1-image-error_{}'.format(val_dataloader.dataset.dataset_name) : L1_error,
            })


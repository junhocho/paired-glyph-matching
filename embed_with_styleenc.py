## python embed_with_styleenc.py --experiment_name bs16

import torch
from torch import nn
from model import CXLoss, DiscriminatorWithClassifier, GeneratorStyle


def load_style_enc(opts):

    generator = GeneratorStyle(n_style=opts.n_style, attr_channel=opts.attr_channel,
                               style_out_channel=opts.style_out_channel,
                               n_res_blocks=opts.n_res_blocks,
                               attention=opts.attention)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    generator = nn.DataParallel(generator)




    import os
    # log_dir = os.path.join("experiments", 'bs-16')
    # log_dir = os.path.join("experiments", 'bs-16')
    # checkpoint_dir = os.path.join(log_dir, "checkpoint")

    # gen_file = gen_file # os.path.join(checkpoint_dir, f"G_10.pth")
    # gen_file = os.path.join(checkpoint_dir, f"G_500.pth")
    # ???

    generator.load_state_dict(torch.load(opts.pretrained))

    # generator.style_enc
    # generator.module.style_enc.infer_style

    style_enc =  generator.module.style_enc
    # randimg = torch.rand([52, 12, 64, 64]).to(device)
    # style_feat = style_enc.infer_style(randimg)  ## torch.Size([52, 128, 1, 1])
    return style_enc

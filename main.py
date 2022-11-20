import datetime
import os
import time

import torch
from torch import nn
from torchvision.utils import save_image

from dataloader import get_loader, get_loader_fontemb
from model import CXLoss, DiscriminatorWithClassifier, GeneratorStyle
from options import get_parser
from vgg_cx import VGG19_CX

# random seed control
import numpy as np
import random

from solver import Solver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb

def train_representation(opts):
    if opts.data_type == "2glyphs":
        # assert(not opts.train_fontcls)
        assert(not opts.train_ae)
        assert(opts.simclr or opts.supcon or opts.train_cae)
        if opts.heads:
            opts.heads = [int(h) for h in opts.heads]
    elif opts.data_type == '1glyph':
        assert(not opts.train_cae)
        assert(not opts.simclr)
        assert(not opts.supcon)
        assert(opts.train_fontcls or opts.train_ae or opts.train_attr)

    # if opts.heads:
    #     assert(opts.data_type=='2glyphs')
    #     opts.heads = [int(h) for h in opts.heads]
    # else:
    #     # without heads, no simclr mode.
    #     assert(opts.data_type=='1glyph')
    #     assert(not opts.simclr)
    print(opts)
    print()
    # Dirs
    log_dir = opts.log_dir
    checkpoint_dir = os.path.join(opts.log_dir, "checkpoint")
    samples_dir = os.path.join(opts.log_dir, "samples")
    logs_dir = os.path.join(opts.log_dir, "logs")


    # Path to data
    # Dataloader
    train_dataloader_list = []
    if opts.data_type == '2glyphs':
        donovan_dataset_name = "donovan_embedding"
        ofl_dataset_name =  "ofl"
        Capitals64_dataset_name =  "Capitals64"
        donovan_attr_path = "./data/explor_all/attributes.txt"
    elif opts.data_type == '1glyph':
        donovan_dataset_name = "donovan_embedding_per_char"
        ofl_dataset_name =  "ofl_per_char"
        Capitals64_dataset_name =  "Capitals64_per_char"
        donovan_attr_path = "./data/explor_all/attributes_alphanumeric.txt"
    elif opts.data_type == 'glyph-set':
        raise NotImplementedError("Not implemented")
    else:
        raise NotImplementedError("Not implemented")
    test_donovan_dataset_name = "donovan_embedding_per_char"
    test_ofl_dataset_name = "ofl_per_char"
    test_Capitals64_dataset_name =  "Capitals64_per_char"
    test_donovan_attr_path ="./data/explor_all/attributes_alphanumeric.txt"


    if 'donovan_embedding' in opts.dataset_name:
        train_dataloader = get_loader_fontemb(
                                    "./data/explor_all/image",
                                    donovan_attr_path,
                                    dataset_name=donovan_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=opts.batch_size, binary=False, mode='train',
                                    augmentation=not opts.no_augmentation, identical_augmentation=opts.identical_augmentation,
                                    unsuper_num=opts.unsuper_num)
        train_dataloader_list.append(train_dataloader)
        test_dataloader = get_loader_fontemb(
                                    "./data/explor_all/image",
                                    test_donovan_attr_path,
                                    dataset_name = test_donovan_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=52,  ## XXX :hardcoded to charnum
                                    mode='test', binary=False)
    elif 'ofl' in opts.dataset_name:
        train_dataloader = get_loader_fontemb(
                                    "./data/ofl_images/",
                                    "./data/ofl_images/glyph_files.txt",
                                    dataset_name=ofl_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=opts.batch_size, binary=False, mode='train',
                                    augmentation=not opts.no_augmentation,
                                    identical_augmentation=opts.identical_augmentation,
                                    num_workers=4)
        train_dataloader_list.append(train_dataloader)
        test_dataloader = get_loader_fontemb(
                                    "./data/ofl_images/",
                                    "./data/ofl_images/glyph_files.txt",
                                    dataset_name = test_ofl_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=52,  ## XXX :hardcoded to charnum
                                    mode='val', binary=False)
    elif 'Capitals64' in opts.dataset_name:
        train_dataloader = get_loader_fontemb(
                                    "./data/Capitals64_split/",
                                    None,
                                    dataset_name=Capitals64_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=opts.batch_size, binary=False, mode='train',
                                    augmentation=not opts.no_augmentation,
                                    identical_augmentation=opts.identical_augmentation,
                                    char_set='capitals',
                                    num_workers=4)
        train_dataloader_list.append(train_dataloader)
        test_dataloader = get_loader_fontemb(
                                    "./data/Capitals64_split/",
                                    None,
                                    dataset_name = test_Capitals64_dataset_name,
                                    image_size=opts.img_size,
                                    batch_size=26,  ## XXX :hardcoded to charnum
                                    char_set='capitals',
                                    mode='val', binary=False)

    if opts.use_wandb:
        wandb.init(project="donovan_embedding")
        wandb.run.name = opts.log_dir.split('experiments/')[-1]
        wandb.config.update(opts)
    # Model
    from network.networks import ResNet34, ResNet18, MLP
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
    print(fontemb_net)
    fontemb_net = fontemb_net.to(device)

    if opts.train_attr:
        attrregressor_net = MLP(opts.feat_dim, opts.attr_channel).to(device)
        print(attrregressor_net)
    else:
        attrregressor_net = None
    num_fonts = len(train_dataloader.dataset)
    num_chars = train_dataloader.dataset.char_num

    if opts.train_fontcls and not opts.supcon:
        fontcls_net = MLP(opts.feat_dim, num_fonts).to(device)
        print(fontcls_net)
    else:
        fontcls_net = None

    if opts.train_charcls:
        charcls_net = MLP(opts.feat_dim, num_chars).to(device)
        print(charcls_net)
    else:
        charcls_net = None

    if opts.train_ae:
        from model import FontDecoder
        fontdec_net = FontDecoder(in_channel=opts.feat_dim, out_channel=3, attention=True).to(device)
        print(fontdec_net)
    elif opts.train_cae:
        from model import FontDecoder
        fontdec_net = FontDecoder(in_channel=opts.feat_dim + num_chars, out_channel=3, attention=True).to(device)
        print(fontdec_net)
    else:
        fontdec_net = None

    models = {'fontemb_net':fontemb_net,
            'attrcls_net':attrregressor_net,
            'fontcls_net':fontcls_net,
            'charcls_net':charcls_net,
            'fontdec_net':fontdec_net
            }


    # Resume training
    if opts.init_epoch > 1:
        orig_checkpoint_dir = os.path.join(opts.orig_log_dir, "checkpoint")
        fontemb_net_file = os.path.join(orig_checkpoint_dir, f"F_{opts.init_epoch}.pth")
        print("Loading and resuem training from: {}".format(fontemb_net_file))
        fontemb_net.load_state_dict(torch.load(fontemb_net_file))
        if opts.train_attr:
            attrregressor_net_file = os.path.join(orig_checkpoint_dir, f"A_{opts.init_epoch}.pth")
            print("Loading and resuem training from: {}".format(attrregressor_net_file))
            attrregressor_net.load_state_dict(torch.load(attrregressor_net_file))
        if opts.train_fontcls and not opts.supcon:
            fontcls_net_file = os.path.join(orig_checkpoint_dir, f"Fontcls_{opts.init_epoch}.pth")
            print("Loading and resuem training from: {}".format(fontcls_net_file))
            fontcls_net.load_state_dict(torch.load(fontcls_net_file))
        if opts.train_charcls:
            charcls_net_file = os.path.join(orig_checkpoint_dir, f"Charcls_{opts.init_epoch}.pth")
            print("Loading and resuem training from: {}".format(charcls_net_file))
            charcls_net.load_state_dict(torch.load(charcls_net_file))
        if opts.train_ae or opts.train_cae:
            fontdec_net_file = os.path.join(orig_checkpoint_dir, f"Fontdec_{opts.init_epoch}.pth")
            print("Loading and resuem training from: {}".format(fontdec_net_file))
            fontdec_net.load_state_dict(torch.load(fontdec_net_file))
        opts.init_epoch += 1   # since this epoch has been already trained.
    if opts.pretrained:
        print("pretrainig from {}".format(opts.pretrained))
        fontemb_net.load_state_dict(torch.load(opts.pretrained))
        # NOTE , Attr, Fontcls, Charcls not loading


    logfile = open(os.path.join(log_dir, "loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    solver = Solver(opts, logfile, val_logfile, device, 
            train_dataloader_list,
            models)
    if opts.pretrained or opts.init_epoch > 1:
        if opts.pretrained:
            step = 0
            epoch = 0
        else:
            step = opts.init_epoch * len(train_dataloader_list[0])
            epoch = opts.init_epoch
        solver.curr_step = step
        solver.evaluate(epoch, test_dataloader)
        print("Pretrained evaluation... :", log_dir)


    for epoch in range(opts.init_epoch, opts.n_epochs+1):
        """
        Train
        """
        batch_idx = 0
        for dataset_i, train_dataloader in enumerate(train_dataloader_list):
            for _, batch in enumerate(train_dataloader):
                solver.train_batch(batch, dataset_i)
                solver.log_results(epoch, batch_idx, dataset_i)
                batch_idx += 1
        """
        Train ends
        """
        if opts.check_freq > 0 and epoch % opts.check_freq == 0:
            solver.evaluate(epoch, test_dataloader)
            print("finishing evaluation... :", log_dir)
        if opts.check_freq > 0 and epoch % opts.check_freq == 0:
            print("save model... :", checkpoint_dir)
            fontemb_net_file = os.path.join(checkpoint_dir, f"F_{epoch}.pth")
            torch.save(fontemb_net.state_dict(), fontemb_net_file)
            if opts.train_attr:
                attrregressor_net_file = os.path.join(checkpoint_dir, f"A_{epoch}.pth")
                torch.save(attrregressor_net.state_dict(), attrregressor_net_file)
            if opts.train_fontcls and not opts.supcon:
                fontcls_net_file = os.path.join(checkpoint_dir, f"Fontcls_{epoch}.pth")
                torch.save(fontcls_net.state_dict(), fontcls_net_file)
            if opts.train_charcls:
                charcls_net_file = os.path.join(checkpoint_dir, f"Charcls_{epoch}.pth")
                torch.save(charcls_net.state_dict(), charcls_net_file)
            if opts.train_ae or opts.train_cae:
                fontdec_net_file = os.path.join(checkpoint_dir, f"Fontdec_{epoch}.pth")
                torch.save(fontdec_net.state_dict(), fontdec_net_file)


"""
Implementing above
"""

def train(opts):
    if opts.use_wandb:
        wandb.init(project='Attr2Font')
        wandb.run.name = opts.experiment_name
        wandb.config.update(opts)
    # Dirs
    log_dir = os.path.join("experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    samples_dir = os.path.join(log_dir, "samples")
    logs_dir = os.path.join(log_dir, "logs")

    # Loss criterion
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_ce = torch.nn.CrossEntropyLoss().to(device)
    criterion_attr = torch.nn.MSELoss().to(device)

    # CX Loss
    if opts.lambda_cx > 0:
        criterion_cx = CXLoss(sigma=0.5).to(device)
        vgg19 = VGG19_CX().to(device)
        vgg19.load_model('vgg19-dcbb9e9d.pth')
        vgg19.eval()
        vgg_layers = ['conv3_3', 'conv4_2']

    # Path to data
    image_dir = os.path.join(opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join(opts.data_root, opts.dataset_name, "attributes.txt")

    # Dataloader
    train_dataloader = get_loader(image_dir, attribute_path,
                                  dataset_name=opts.dataset_name,
                                  image_size=opts.img_size,
                                  n_style=opts.n_style,
                                  batch_size=opts.batch_size, binary=False)
    val_dataloader = get_loader(image_dir, attribute_path,
                                 dataset_name=opts.dataset_name,
                                 image_size=opts.img_size,
                                 n_style=opts.n_style, batch_size=8,
                                 mode='test', binary=False)

    test_dataloader = get_loader(image_dir, attribute_path,
                                 dataset_name=opts.dataset_name,
                                 image_size=opts.img_size,
                                 n_style=opts.n_style, batch_size=52,
                                 mode='test', binary=False)

    # Model
    if opts.exterial_style_enc_down:
        assert(opts.feat_dim == opts.style_out_channel)
        assert(opts.n_style == 1)
        from network.networks import ResNet34, ResNet18, MLP
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
        if opts.load_style_enc:
            fontemb_net.load_state_dict(torch.load(opts.load_style_enc))
        down = fontemb_net
    else:
        down = None

    generator = GeneratorStyle(n_style=opts.n_style, attr_channel=opts.attr_channel,
                               style_out_channel=opts.style_out_channel,
                               n_res_blocks=opts.n_res_blocks,
                               attention=opts.attention, down=down, train_down=not opts.cut_grad_styl_enc_down)
    print(generator)
    discriminator = DiscriminatorWithClassifier()
    # Attrbute embedding
    # attribute: N x 37 -> N x 37 x 64
    attribute_embed = nn.Embedding(opts.attr_channel, opts.attr_embed)
    # unsupervise font num + 1 dummy id (for supervise)
    attr_unsuper_tolearn = nn.Embedding(opts.unsuper_num+1, opts.attr_channel)  # attribute intensity

    if opts.multi_gpu:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        attribute_embed = nn.DataParallel(attribute_embed)
        attr_unsuper_tolearn = nn.DataParallel(attr_unsuper_tolearn)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    attribute_embed = attribute_embed.to(device)
    attr_unsuper_tolearn = attr_unsuper_tolearn.to(device)

    # Discriminator output patch shape
    patch = (1, opts.img_size // 2**4, opts.img_size // 2**4)

    # optimizers
    optimizer_G = torch.optim.Adam([
        {'params': generator.parameters()},
        {'params': attr_unsuper_tolearn.parameters(), 'lr': 1e-3},
        {'params': attribute_embed.parameters(), 'lr': 1e-3}],
        lr=opts.lr, betas=(opts.b1, opts.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, betas=(opts.b1, opts.b2))

    # Resume training
    if opts.init_epoch > 1:
        gen_file = os.path.join(checkpoint_dir, f"G_{opts.init_epoch}.pth")
        attr_unsuper_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{opts.init_epoch}.pth")
        attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{opts.init_epoch}")
        dis_file = os.path.join(checkpoint_dir, f"D_{opts.init_epoch}.pth")

        generator.load_state_dict(torch.load(gen_file))
        attr_unsuper_tolearn.load_state_dict(torch.load(attr_unsuper_file))
        attribute_embed.load_state_dict(torch.load(attribute_embed_file))
        discriminator.load_state_dict(torch.load(dis_file))

    prev_time = time.time()
    logfile = open(os.path.join(log_dir, "loss_log.txt"), 'w')
    val_logfile = open(os.path.join(log_dir, "val_loss_log.txt"), 'w')

    attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
    attrid = attrid.view(1, attrid.size(0))
    attrid = attrid.repeat(opts.batch_size, 1)

    for epoch in range(opts.init_epoch, opts.n_epochs+1):
        for batch_idx, batch in enumerate(train_dataloader):
            img_A = batch['img_A'].to(device)
            attr_A_data = batch['attr_A'].to(device)
            fontembd_A = batch['fontembed_A'].to(device)
            label_A = batch['label_A'].to(device)
            charclass_A = batch['charclass_A'].to(device)
            styles_A = batch['styles_A'].to(device)

            img_B = batch['img_B'].to(device)
            attr_B_data = batch['attr_B'].to(device)
            fontembd_B = batch['fontembed_B'].to(device)
            label_B = batch['label_B'].to(device)
            charclass_B = batch['charclass_B'].to(device)
            styles_B = batch['styles_B'].to(device)

            valid = torch.ones((img_A.size(0), *patch)).to(device)
            fake = torch.zeros((img_A.size(0), *patch)).to(device)

            # Construct attribute
            attr_raw_A = attribute_embed(attrid)
            attr_raw_B = attribute_embed(attrid)

            attr_A_embd = attr_unsuper_tolearn(fontembd_A)
            attr_A_embd = attr_A_embd.view(attr_A_embd.size(0), attr_A_embd.size(2))
            attr_A_embd = torch.sigmoid(3*attr_A_embd)  # convert to [0, 1]
            attr_A_intensity = label_A * attr_A_data + (1 - label_A) * attr_A_embd

            attr_A_intensity_u = attr_A_intensity.unsqueeze(-1)
            attr_A = attr_A_intensity_u * attr_raw_A

            attr_B_embd = attr_unsuper_tolearn(fontembd_B)
            attr_B_embd = attr_B_embd.view(attr_B_embd.size(0), attr_B_embd.size(2))
            attr_B_embd = torch.sigmoid(3*attr_B_embd)  # convert to [0, 1]
            attr_B_intensity = label_B * attr_B_data + (1 - label_B) * attr_B_embd

            attr_B_intensity_u = attr_B_intensity.unsqueeze(-1)
            attr_B = attr_B_intensity_u * attr_raw_B

            delta_intensity = attr_B_intensity - attr_A_intensity
            delta_attr = attr_B - attr_A

            # Forward G and D
            if opts.attr_zero:
                delta_intensity = torch.zeros_like(delta_intensity)
                delta_attr = torch.zeros_like(delta_attr)
                attr_B_intensity = torch.zeros_like(attr_B_intensity)
                attr_A_intensity = torch.zeros_like(attr_A_intensity)
                fake_B, content_logits_A = generator(img_A, styles_B, delta_intensity, delta_attr)
            elif opts.style_zero:
                styles_A = torch.zeros_like(styles_A)
                fake_B, content_logits_A = generator(img_A, styles_A, delta_intensity, delta_attr)
            else:
                fake_B, content_logits_A = generator(img_A, styles_A, delta_intensity, delta_attr)

            pred_fake, real_A_attr_fake, fake_B_attr_fake = discriminator(img_A, fake_B, charclass_B, attr_B_intensity)

            if opts.lambda_cx > 0:
                vgg_fake_B = vgg19(fake_B)
                vgg_img_B = vgg19(img_B)

            # Calculate losses
            loss_GAN = opts.lambda_GAN * criterion_GAN(pred_fake, valid)
            loss_pixel = opts.lambda_l1 * criterion_pixel(fake_B, img_B)

            loss_char_A = criterion_ce(content_logits_A, charclass_A.view(charclass_A.size(0)))  # +
            loss_char_A = opts.lambda_char * loss_char_A

            loss_attr = torch.zeros(1).to(device)
            if opts.dis_pred and not opts.attr_zero:
                loss_attr += opts.lambda_attr * criterion_attr(attr_A_intensity, real_A_attr_fake)
                loss_attr += opts.lambda_attr * criterion_attr(attr_B_intensity, fake_B_attr_fake)

            # CX loss
            loss_CX = torch.zeros(1).to(device)
            if opts.lambda_cx > 0:
                for l in vgg_layers:
                    cx = criterion_cx(vgg_img_B[l], vgg_fake_B[l])
                    loss_CX += cx * opts.lambda_cx

            loss_G = loss_GAN + loss_pixel + loss_char_A + loss_CX + loss_attr

            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # Forward D
            pred_real, A_attr_real, B_attr_real = discriminator(img_A, img_B, charclass_B, attr_B_intensity.detach())
            loss_real = criterion_GAN(pred_real, valid)

            loss_attr_D = torch.zeros(1).to(device)
            if opts.dis_pred and not opts.attr_zero:
                loss_attr_D += criterion_attr(attr_A_intensity.detach(), A_attr_real)
                loss_attr_D += criterion_attr(attr_B_intensity.detach(), B_attr_real)

            pred_fake, A_attr_fake, B_attr_fake = discriminator(img_A, fake_B.detach(), charclass_B, attr_B_intensity.detach())  # noqa
            if opts.dis_pred and not opts.attr_zero:
                loss_attr_D += criterion_attr(attr_A_intensity.detach(), A_attr_fake)
                loss_attr_D += criterion_attr(attr_B_intensity.detach(), B_attr_fake)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = loss_real + loss_fake + loss_attr_D

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            batches_done = (epoch - opts.init_epoch) * len(train_dataloader) + batch_idx
            batches_left = (opts.n_epochs - opts.init_epoch) * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left*(time.time() - prev_time))
            prev_time = time.time()

            message = (
                f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, ETA: {time_left}, "
                f"D loss: {loss_D.item():.6f}, G loss: {loss_G.item():.6f}, "
                f"loss_pixel: {loss_pixel.item():.6f}, "
                f"loss_adv: {loss_GAN.item():.6f}, "
                f"loss_char_A: {loss_char_A.item():.6f}, "
                f"loss_CX: {loss_CX.item():.6f}, "
                f"loss_attr: {loss_attr.item(): .6f}"
            )
            if opts.use_wandb:
                wandb.log({
                    "step" : (epoch - 1)*len(train_dataloader) + batch_idx + 1,
                    "D_loss" : loss_D.item(),
                    "G_loss" : loss_G.item(),
                    "loss_pixel" : loss_pixel.item(),
                    "loss_adv" : loss_GAN.item(), 
                    "loss_char_A": loss_char_A.item(), 
                    "loss_CX": loss_CX.item(), 
                    "loss_attr": loss_attr.item()
                    })

            print(message)
            logfile.write(message + '\n')
            logfile.flush()

            if batches_done % opts.log_freq == 0:
                img_sample = torch.cat((img_A.data, fake_B.data, img_B.data), -2)
                save_file = os.path.join(logs_dir, f"epoch_{epoch}_batch_{batches_done}.png")
                save_image(img_sample, save_file, nrow=8, normalize=True)

            if batches_done % opts.sample_freq == 0:
                with torch.no_grad():
                    val_attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
                    val_attrid = val_attrid.repeat(8, 1)
                    val_l1_loss = torch.zeros(1).to(device)
                    for val_idx, val_batch in enumerate(val_dataloader):
                        if val_idx == 20:  # only validate on first 20 batches, you can change it
                            break
                        val_img_A = val_batch['img_A'].to(device)
                        val_fontembed_A = val_batch['fontembed_A'].to(device)
                        val_styles_A = val_batch['styles_A'].to(device)
                        val_styles_B = val_batch['styles_B'].to(device)

                        val_img_B = val_batch['img_B'].to(device)

                        val_attr_A_intensity = attr_unsuper_tolearn(val_fontembed_A)
                        val_attr_A_intensity = val_attr_A_intensity.view(val_attr_A_intensity.size(0), val_attr_A_intensity.size(2))
                        val_attr_A_intensity = torch.sigmoid(3*val_attr_A_intensity)  # convert to [0, 1]

                        val_attr_B_intensity = val_batch['attr_B'].to(device)

                        val_attr_raw_A = attribute_embed(val_attrid)
                        val_attr_raw_B = attribute_embed(val_attrid)

                        val_intensity_A_u = val_attr_A_intensity.unsqueeze(-1)
                        val_intensity_B_u = val_attr_B_intensity.unsqueeze(-1)

                        val_attr_A = val_intensity_A_u * val_attr_raw_A
                        val_attr_B = val_intensity_B_u * val_attr_raw_B

                        val_intensity = val_attr_B_intensity - val_attr_A_intensity
                        val_attr = val_attr_B - val_attr_A

                        if opts.attr_zero:
                            val_intensity = torch.zeros_like(val_intensity)
                            val_attr = torch.zeros_like(val_attr)
                            val_fake_B, _ = generator(val_img_A, val_styles_B, val_intensity, val_attr)
                        elif opts.style_zero:
                            val_styles_A = torch.zeros_like(val_styles_A)
                            val_fake_B, _ = generator(val_img_A, val_styles_A, val_intensity, val_attr)
                        else:
                            val_fake_B, _ = generator(val_img_A, val_styles_A, val_intensity, val_attr)

                        val_l1_loss += criterion_pixel(val_fake_B, val_img_B)

                        img_sample = torch.cat((val_img_A.data, val_fake_B.data, val_img_B.data), -2)
                        save_file = os.path.join(samples_dir, f"epoch_{epoch}_idx_{val_idx}.png")
                        save_image(img_sample, save_file, nrow=8, normalize=True)

                    val_l1_loss = val_l1_loss / 20
                    val_msg = (
                        f"Epoch: {epoch}/{opts.n_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, "
                        f"L1: {val_l1_loss.item(): .6f}"
                    )
                    if opts.use_wandb:
                        wandb.log({
                            "epoch" : epoch,
                            "val_L1" : val_l1_loss.item()
                            })
                    val_logfile.write(val_msg + "\n")
                    val_logfile.flush()

        if opts.check_freq > 0 and epoch % opts.check_freq == 0:
            gen_file_file = os.path.join(checkpoint_dir, f"G_{epoch}.pth")
            attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{epoch}.pth")
            attr_unsuper_embed_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{epoch}.pth")
            dis_file_file = os.path.join(checkpoint_dir, f"D_{epoch}.pth")

            torch.save(generator.state_dict(), gen_file_file)
            torch.save(attribute_embed.state_dict(), attribute_embed_file)
            torch.save(attr_unsuper_tolearn.state_dict(), attr_unsuper_embed_file)
            torch.save(discriminator.state_dict(), dis_file_file)

            test_logfile = open(os.path.join(log_dir, f"test_loss_log_{opts.test_epoch}.txt"), 'w')
            results_dir = os.path.join(log_dir, "results")
            generator.eval()
            test_one_epoch(opts, test_logfile, epoch,
                           checkpoint_dir, results_dir,
                           generator, attribute_embed, attr_unsuper_tolearn,
                           test_dataloader, criterion_pixel, use_wandb=opts.use_wandb)
            generator.train()


def test_one_epoch(opts, test_logfile, test_epoch,
                   checkpoint_dir, results_dir,
                   generator, attribute_embed,
                   attr_unsuper_tolearn, test_dataloader,
                   criterion_pixel, use_wandb):
    print(f"Testing epoch: {test_epoch}")

    gen_file = os.path.join(checkpoint_dir, f"G_{test_epoch}.pth")
    attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{test_epoch}.pth")
    attr_unsuper_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{test_epoch}.pth")

    generator.load_state_dict(torch.load(gen_file))
    attribute_embed.load_state_dict(torch.load(attribute_embed_file))
    attr_unsuper_tolearn.load_state_dict(torch.load(attr_unsuper_file))

    with torch.no_grad():
        test_attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
        test_attrid = test_attrid.repeat(52, 1)
        test_l1_loss = torch.zeros(1).to(device)

        for test_idx, test_batch in enumerate(test_dataloader):
            test_img_A = test_batch['img_A'].to(device)
            test_fontembed_A = test_batch['fontembed_A'].to(device)
            test_styles_A = test_batch['styles_A'].to(device)

            test_img_B = test_batch['img_B'].to(device)

            test_attr_A_intensity = attr_unsuper_tolearn(test_fontembed_A)
            test_attr_A_intensity = test_attr_A_intensity.view(test_attr_A_intensity.size(0), test_attr_A_intensity.size(2))  # noqa
            test_attr_A_intensity = torch.sigmoid(3*test_attr_A_intensity)  # convert to [0, 1]

            test_attr_B_intensity = test_batch['attr_B'].to(device)

            test_attr_raw_A = attribute_embed(test_attrid)
            test_attr_raw_B = attribute_embed(test_attrid)

            test_intensity_A_u = test_attr_A_intensity.unsqueeze(-1)
            test_intensity_B_u = test_attr_B_intensity.unsqueeze(-1)

            test_attr_A = test_intensity_A_u * test_attr_raw_A
            test_attr_B = test_intensity_B_u * test_attr_raw_B

            test_intensity = test_attr_B_intensity - test_attr_A_intensity
            test_attr = test_attr_B - test_attr_A

            if opts.attr_zero:
                test_intensity = torch.zeros_like(test_intensity)
                test_attr = torch.zeros_like(test_attr)
                # val_fake_B, _ = generator(val_img_A, val_styles_B, val_intensity, val_attr)
                ## XXX cannot test
                test_fake_B, _ = generator(test_img_A, test_styles_B, test_intensity, test_attr)
            else:
                test_fake_B, _ = generator(test_img_A, test_styles_A, test_intensity, test_attr)
            test_l1_loss += criterion_pixel(test_fake_B, test_img_B)

            img_sample = torch.cat((test_img_A.data, test_fake_B.data, test_img_B.data), -2)
            save_file = os.path.join(results_dir, f"test_{test_epoch}_idx_{test_idx}.png")
            save_image(img_sample, save_file, nrow=52, normalize=True)

        test_l1_loss = test_l1_loss / len(test_dataloader)
        if use_wandb:
            wandb.log({
                "epoch" : test_epoch,
                "L1-error" : test_l1_loss.item()
                })

        test_msg = (
            f"Epoch: {test_epoch}/{opts.n_epochs}, "
            f"L1: {test_l1_loss.item(): .6f}"
        )
        print(test_msg)
        test_logfile.write(test_msg + "\n")
        test_logfile.flush()


def test(opts):
    if opts.use_wandb:
        wandb.init(project="Attr2Font")
        wandb.run.name = opts.experiment_name
        wandb.config.update(opts)
    # Dirs
    log_dir = os.path.join("experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    results_dir = os.path.join(log_dir, "results")

    # Path to data
    image_dir = os.path.join(opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join(opts.data_root, opts.dataset_name, "attributes.txt")

    test_dataloader = get_loader(image_dir, attribute_path,
                                 dataset_name=opts.dataset_name,
                                 image_size=opts.img_size,
                                 n_style=opts.n_style, batch_size=52,
                                 mode='test', binary=False)

    # Model
    criterion_pixel = torch.nn.L1Loss().to(device)
    generator = GeneratorStyle(n_style=opts.n_style, attr_channel=opts.attr_channel,
                               style_out_channel=opts.style_out_channel,
                               n_res_blocks=opts.n_res_blocks,
                               attention=opts.attention)
    # Attrbute embedding
    # attribute: N x 37 -> N x 37 x 64
    attribute_embed = nn.Embedding(opts.attr_channel, opts.attr_embed)
    # unsupervise font num + 1 dummy id (for supervise)
    attr_unsuper_tolearn = nn.Embedding(opts.unsuper_num+1, opts.attr_channel)  # attribute intensity

    if opts.multi_gpu:
        generator = nn.DataParallel(generator)
        attribute_embed = nn.DataParallel(attribute_embed)
        attr_unsuper_tolearn = nn.DataParallel(attr_unsuper_tolearn)

    generator = generator.to(device)
    attribute_embed = attribute_embed.to(device)
    attr_unsuper_tolearn = attr_unsuper_tolearn.to(device)

    test_logfile = open(os.path.join(log_dir, f"test_loss_log_{opts.test_epoch}.txt"), 'w')

    generator.eval()
    if opts.test_epoch == 0:
        for test_epoch in range(opts.check_freq, opts.n_epochs+1, opts.check_freq):
            test_one_epoch(opts, test_logfile, test_epoch,
                           checkpoint_dir, results_dir,
                           generator, attribute_embed, attr_unsuper_tolearn,
                           test_dataloader, criterion_pixel, use_wandb=opts.use_wandb)
    else:
        test_one_epoch(opts, test_logfile, opts.test_epoch,
                       checkpoint_dir, results_dir,
                       generator, attribute_embed, attr_unsuper_tolearn,
                       test_dataloader, criterion_pixel)


def interp(opts):
    # Dirs
    log_dir = os.path.join("experiments", opts.experiment_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    results_dir = os.path.join(log_dir, "interps")

    # Path to data
    image_dir = os.path.join(opts.data_root, opts.dataset_name, "image")
    attribute_path = os.path.join(opts.data_root, opts.dataset_name, "attributes.txt")

    test_dataloader = get_loader(image_dir, attribute_path,
                                 dataset_name=opts.dataset_name,
                                 image_size=opts.img_size,
                                 n_style=opts.n_style, batch_size=52,
                                 mode='test', binary=False)

    # Model
    generator = GeneratorStyle(n_style=opts.n_style, attr_channel=opts.attr_channel,
                               style_out_channel=opts.style_out_channel,
                               n_res_blocks=opts.n_res_blocks,
                               attention=opts.attention)
    # Attrbute embedding
    # attribute: N x 37 -> N x 37 x 64
    attribute_embed = nn.Embedding(opts.attr_channel, opts.attr_embed)
    # unsupervise font num + 1 dummy id (for supervise)
    attr_unsuper_tolearn = nn.Embedding(opts.unsuper_num+1, opts.attr_channel)  # attribute intensity

    assert opts.test_epoch > 0 and opts.test_epoch % opts.check_freq == 0, "Please choose correct test epoch"

    if opts.multi_gpu:
        generator = nn.DataParallel(generator)
        attribute_embed = nn.DataParallel(attribute_embed)
        attr_unsuper_tolearn = nn.DataParallel(attr_unsuper_tolearn)

    generator = generator.to(device)
    attribute_embed = attribute_embed.to(device)
    attr_unsuper_tolearn = attr_unsuper_tolearn.to(device)

    print(f"Interpolating epoch: {opts.test_epoch}")

    gen_file = os.path.join(checkpoint_dir, f"G_{opts.test_epoch}.pth")
    attribute_embed_file = os.path.join(checkpoint_dir, f"attribute_embed_{opts.test_epoch}.pth")
    attr_unsuper_file = os.path.join(checkpoint_dir, f"attr_unsuper_embed_{opts.test_epoch}.pth")

    generator.load_state_dict(torch.load(gen_file))
    attribute_embed.load_state_dict(torch.load(attribute_embed_file))
    attr_unsuper_tolearn.load_state_dict(torch.load(attr_unsuper_file))

    with torch.no_grad():
        test_attrid = torch.tensor([i for i in range(opts.attr_channel)]).to(device)
        test_attrid = test_attrid.repeat(52, 1)  # 52 is char number
        for test_idx, test_batch in enumerate(test_dataloader):
            test_img_A = test_batch['img_A'].to(device)
            test_fontembed_A = test_batch['fontembed_A'].to(device)
            test_styles_A = test_batch['styles_A'].to(device)

            test_img_B = test_batch['img_B'].to(device)

            test_attr_A_intensity = attr_unsuper_tolearn(test_fontembed_A)
            test_attr_A_intensity = test_attr_A_intensity.view(test_attr_A_intensity.size(0), test_attr_A_intensity.size(2))  # noqa
            test_attr_A_intensity = torch.sigmoid(3*test_attr_A_intensity)  # convert to [0, 1]

            test_attr_B_intensity = test_batch['attr_B'].to(device)

            test_attr_raw_A = attribute_embed(test_attrid)
            test_attr_raw_B = attribute_embed(test_attrid)

            test_intensity_A_u = test_attr_A_intensity.unsqueeze(-1)
            test_intensity_B_u = test_attr_B_intensity.unsqueeze(-1)

            test_attr_A = test_intensity_A_u * test_attr_raw_A
            test_attr_B = test_intensity_B_u * test_attr_raw_B

            test_intensity = test_attr_B_intensity - test_attr_A_intensity
            test_attr = test_attr_B - test_attr_A

            print(f"interp batch idx {test_idx}")

            # All attributes interpolation
            img_sample = [test_img_A.data]
            for alpha in range(opts.interp_cnt):
                alpha /= opts.interp_cnt - 1
                test_alpha_intesnsity = alpha * test_intensity.clone().detach()
                test_alpha_attr = alpha * test_attr.clone().detach()
                test_fake_B_alpha, _ = generator(test_img_A.clone().detach(), test_styles_A.clone().detach(),
                                                 test_alpha_intesnsity, test_alpha_attr)
                img_sample.append(test_fake_B_alpha.data)
            img_sample.append(test_img_B.data)
            img_sample = torch.cat(img_sample, -2)

            results_dir = os.path.join(log_dir, "results")
            save_file = os.path.join(results_dir, f"interp_batch_{opts.test_epoch}_idx_{test_idx}.png")
            save_image(img_sample, save_file, nrow=52, normalize=True, padding=0)

            # Random all char same
            img_sample_random_attr = [test_img_A.data]
            for alpha in range(opts.interp_cnt):
                alpha /= opts.interp_cnt - 1
                one_batch_random = torch.rand_like(test_attr_B_intensity[0]).unsqueeze(0).to(device)

                test_intensity_B_beta = one_batch_random.repeat(52, 1).to(device)
                test_intensity_B_beta_u = test_intensity_B_beta.unsqueeze(-1)
                test_attr_B_beta = test_intensity_B_beta_u * test_attr_raw_B.clone().detach()
                test_intensity_beta = test_intensity_B_beta - test_attr_A_intensity.clone().detach()
                test_attr_beta = test_attr_B_beta - test_attr_A.clone().detach()
                test_fake_B_beta, _ = generator(test_img_A.clone().detach(), test_styles_A.clone().detach(),
                                                test_intensity_beta, test_attr_beta)

                img_sample_random_attr.append(test_fake_B_beta.data)
            img_sample_random_attr.append(test_img_B.data)
            img_sample_random_attr = torch.cat(img_sample_random_attr, -2)

            save_file_sp = os.path.join(results_dir, f"all_char_same_specific_attr_random_{opts.test_epoch}_idx_{test_idx}.png")  # noqa
            save_image(img_sample_random_attr, save_file_sp, nrow=52, normalize=True, padding=0)

            # Specific attribute source
            for attr_idx in range(opts.attr_channel):
                img_sample_specific_attr = [test_img_A.data]
                for alpha in range(opts.interp_cnt):
                    alpha /= opts.interp_cnt - 1
                    test_intensity_B_beta = test_attr_A_intensity.clone().detach()
                    test_intensity_B_beta[:, attr_idx] = alpha
                    test_intensity_B_beta_u = test_intensity_B_beta.unsqueeze(-1)
                    test_attr_B_beta = test_intensity_B_beta_u * test_attr_raw_B.clone().detach()
                    test_intensity_beta = test_intensity_B_beta - test_attr_A_intensity.clone().detach()
                    test_attr_beta = test_attr_B_beta - test_attr_A.clone().detach()
                    test_fake_B_beta, _ = generator(test_img_A.clone().detach(), test_styles_A.clone().detach(),
                                                    test_intensity_beta, test_attr_beta)

                    img_sample_specific_attr.append(test_fake_B_beta.data)
                img_sample_specific_attr = torch.cat(img_sample_specific_attr, -2)

                save_file_sp = os.path.join(results_dir, f"specific_attr_source_{attr_idx}_{opts.test_epoch}_idx_{test_idx}.png")  # noqa
                save_image(img_sample_specific_attr, save_file_sp, nrow=52, normalize=True, padding=0)

            # Specific attribute target
            for attr_idx in range(opts.attr_channel):
                # img_sample_specific_attr = [test_img_A.data]
                img_sample_specific_attr = []
                for alpha in range(opts.interp_cnt):
                    alpha /= opts.interp_cnt - 1
                    test_intensity_B_beta = test_attr_B_intensity.clone().detach()
                    test_intensity_B_beta[:, attr_idx] = alpha
                    test_intensity_B_beta_u = test_intensity_B_beta.unsqueeze(-1)
                    test_attr_B_beta = test_intensity_B_beta_u * test_attr_raw_B.clone().detach()
                    test_intensity_beta = test_intensity_B_beta - test_attr_A_intensity.clone().detach()
                    test_attr_beta = test_attr_B_beta - test_attr_A.clone().detach()
                    test_fake_B_beta, _ = generator(test_img_A.clone().detach(), test_styles_A.clone().detach(),
                                                    test_intensity_beta, test_attr_beta)
                    img_sample_specific_attr.append(test_fake_B_beta.data)
                img_sample_specific_attr.append(test_img_B.data)
                img_sample_specific_attr = torch.cat(img_sample_specific_attr, -2)

                save_file_sp = os.path.join(results_dir, f"specific_attr_target_{attr_idx}_{opts.test_epoch}_idx_{test_idx}.png")  # noqa
                save_image(img_sample_specific_attr, save_file_sp, nrow=52, normalize=True, padding=0)


def main():
    parser = get_parser()
    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    os.makedirs("experiments", exist_ok=True)

    if opts.phase == 'train':
        # Create directories
        log_dir = os.path.join("experiments", opts.experiment_name)
        os.makedirs(log_dir, exist_ok=False)  # False to prevent multiple train run by mistake
        os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoint"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "interps"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)

        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(log_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train(opts)
    elif opts.phase == 'train-representation':
        # Create directories
        log_dir = os.path.join("experiments", opts.experiment_name)
        ## now do not makedir when exist
        # if os.path.exists(log_dir) and opts.init_epoch == 1:  # if scratch training, then mark experiment with time
        #     log_dir += '-'
        #     log_dir += datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if opts.init_epoch > 1:
            opts.orig_log_dir = log_dir  # this path will be used for resuming model.
            # log_dir += "-resumed"
        else:
            os.makedirs(log_dir, exist_ok=False)  # False to prevent multiple train run by mistake
        opts.log_dir = log_dir

        os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoint"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "interps"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)

        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(log_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        train_representation(opts)
    elif opts.phase == 'test':
        print(f"Testing on experiment {opts.experiment_name}...")
        test(opts)
    elif opts.phase == 'test_interp':
        print(f"Testing interpolation on experiment {opts.experiment_name}...")
        interp(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()

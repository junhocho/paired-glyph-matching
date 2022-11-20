import torch
import wandb
import time
import datetime
from torch import nn
import torch.nn.functional as F
from evaluation import retrieval_evaluation, QR_evaluation, attribute_evaluation, L1_gen_evaluation

class Solver(object):
    def __init__(self, opts, logfile, val_logfile, device, train_dataloader_list,
            models):
        self.opts = opts
        self.logfile  = logfile
        self.val_logfile = val_logfile
        self.device = device
        self.num_dataset = len(train_dataloader_list)
        self.num_steps_per_epoch = sum([len(tt) for tt in train_dataloader_list])
        self.char_num = train_dataloader_list[0].dataset.char_num


        self.fontemb_net = models['fontemb_net']
        self.attrregressor_net = models['attrcls_net']
        self.fontcls_net = models['fontcls_net']
        self.charcls_net = models['charcls_net']
        self.fontdec_net = models['fontdec_net']

        # Loss criterion_sim
        if opts.simclr:
            from network.nt_xent import NT_Xent
            # see appendix B.7.: temperature under different batch sizes
            self.criterion_sim = NT_Xent(opts.batch_size, opts.temperature, 1).to(device)
        elif opts.supcon:
            from network.losses import SupConLoss
            # see appendix B.7.: temperature under different batch sizes
            self.criterion_sim = SupConLoss(temperature=opts.temperature).to(device)
        else:
            from network.losses import PUILoss
            self.criterion_sim = PUILoss(device, lamda=2.0)

        if opts.train_fontcls or opts.train_charcls and not opts.supcon:
            self.criterion_ce = torch.nn.CrossEntropyLoss().to(device)  ## used for font-classification / char-classification

        if opts.train_ae or opts.train_cae:
            self.criterion_pixel = torch.nn.L1Loss().to(device)

        if opts.train_attr:
            attr_loss_type = "BCE"
            if attr_loss_type == "MSE":
                self.criterion_attr = torch.nn.MSELoss(reduction = 'none').to(device)
                def attr_logit(x):
                    return x
            elif attr_loss_type == "BCE":
                self.criterion_attr = torch.nn.BCEWithLogitsLoss(reduction = 'none').to(device)
                sigmoid = nn.Sigmoid().to(device)
                sigmoid.eval()
                def attr_logit(x):
                    return sigmoid(x)
            self.attr_logit = attr_logit
            self.criterion_attr_eval = torch.nn.MSELoss(reduction = 'none').to(device)
            self.criterion_attr_eval.eval()

        # optimizers
        params = [{"params": m.parameters()} for k,m in models.items() if m]
        # params = [{"params": m.parameters()} for k,m in models.items() if not k in ['fontemb_net', 'fontcls_net'] and m ]
        # params += [{"params": self.fontemb_net.parameters(), "lr":opts.lr_fontembcls}]
        self.optimizer = torch.optim.Adam(params, lr=opts.lr, betas=(opts.b1, opts.b2))



        self.best_ret_accuracy = 0

    def train_batch(self, batch, dataset_i):
        self.prev_time = time.time()
        opts = self.opts

        image, charclass, fontclass, attr_data, label = self.load_batch(batch) ## image, charclass could be tuple

        feature = self.train_fontemb(image, charclass, fontclass,  dataset_i)
        if opts.train_attr:
            self.train_attr(feature, attr_data, label)
        # backward
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def load_batch(self, batch):
        device = self.device 
        if self.opts.data_type == '2glyphs':
            img_i = batch['img_i'].to(device)
            charclass_i = batch['charclass_i'].to(device)
            img_j = batch['img_j'].to(device)
            charclass_j = batch['charclass_j'].to(device)
            image = (img_i, img_j)
            charclass = (charclass_i, charclass_j)
        elif self.opts.data_type == '1glyph':
            image = batch['img_i'].to(device)
            charclass = batch['charclass_i'].to(device)

        fontclass = batch['fontclass'].to(device)
        attr_data = batch['attr'].to(device)
        self.label = batch['label_A'].to(device)

        return image, charclass, fontclass, attr_data, self.label

    def train_fontemb(self, image, charclass, fontclass, dataset_i):
        """
        forward pass of fontemb and compute loss
        """
        if self.opts.data_type == '2glyphs':
            img_i = image[0]
            img_j = image[1]
            charclass_i = charclass[0]
            charclass_j = charclass[1]

            loss = torch.zeros(1).to(self.device)
            self.loss_dict = {}
            ## infer i
            feat_i, output_i = self.fontemb_net(img_i)

            if self.opts.simclr or self.opts.supcon:
                """
                This trains model with contrastivel learning
                """
                ## model forward & backward
                ## infer j
                feat_j, output_j = self.fontemb_net(img_j)
                feature = (feat_i, feat_j)

                z_i = output_i[dataset_i] # dataset_i used as index of head
                z_j = output_j[dataset_i] # if opts.simclr: use simclr projection, else: use pui softmax head

                if self.opts.simclr:
                    nt_xent_loss = self.criterion_sim(z_i, z_j)
                else:
                    z_i = F.normalize(z_i, dim=1)
                    z_j = F.normalize(z_j, dim=1)
                    features = torch.cat([z_i.unsqueeze(1), z_j.unsqueeze(1)], dim=1)

                    if self.opts.train_fontcls:
                        print("fontcls for supcon but not effective.")
                        nt_xent_loss = self.criterion_sim(features, fontclass.view(-1))
                    else:
                        nt_xent_loss = self.criterion_sim(features)
                self.loss_dict["loss_nt_xent"] = nt_xent_loss
                loss += nt_xent_loss

            if self.opts.train_cae:
                feature = (feat_i, None)
                if self.opts.train_fontcls:
                    font_out = self.fontcls_net(feat_i)
                    cls_loss  = self.criterion_ce(font_out, fontclass.view(-1)) * self.opts.lambda_fontcls
                    self.loss_dict["loss_cls"] = cls_loss
                    loss += cls_loss
                # feat_i, _ = self.fontemb_net(img_i)
                if self.opts.grad_cut_fontdec:
                    fontdec_input = feat_i.detach()
                else:
                    fontdec_input = feat_i
                char_num = self.char_num
                one_hot = torch.nn.functional.one_hot(charclass_j, num_classes=char_num).to(self.device)  # torch.Size([64, 1, 52])
                feat_char_cat = torch.cat([fontdec_input, one_hot.squeeze(1)], dim=1)
                out = self.fontdec_net(feat_char_cat)

                pixel_loss = self.criterion_pixel(out, img_j)
                self.loss_dict["loss_pixel"] = pixel_loss
                loss += pixel_loss

            self.loss_dict["loss"] = loss

        elif self.opts.data_type == '1glyph':
            """
            This mode classifies or autoencode font. Technically, can be joint trained but we dont try
            """
            if self.opts.train_fontcls:
                feature, _ = self.fontemb_net(image)
                font_out = self.fontcls_net(feature)


                loss = self.criterion_ce(font_out, fontclass.view(-1)) * self.opts.lambda_fontcls
                self.loss_dict = {"loss": loss}

            if self.opts.train_ae:
                """
                This mode classifies or autoencode font.
                """
                # image_feature = self.fontemb_net.image_encode(image)
                # out, feature = self.fontdec_net(image_feature)

                feature, _ = self.fontemb_net(image)
                out = self.fontdec_net(feature)

                loss = self.criterion_pixel(out, image)
                self.loss_dict = {"loss": loss}

            if not self.opts.train_fontcls and not self.opts.train_ae and self.opts.train_attr:
                '''
                train attribute only
                '''
                feature, _ = self.fontemb_net(image)
                loss = torch.zeros(1).to(self.device)
                self.loss_dict = {"loss": loss}

        self.loss = self.loss_dict['loss']
        return feature

    def train_attr(self, feature, attr_data, label):
        # if self.opts.simclr or self.opts.supcon:
        if self.opts.data_type == '2glyphs':
            feat_i = feature[0]
            feat_j = feature[1]
            attr_i = self.attrregressor_net(feat_i)
            attr_j = self.attrregressor_net(feat_j)
            loss_i = self.criterion_attr(attr_i, attr_data)
            loss_j = self.criterion_attr(attr_j, attr_data)
            loss_attr = loss_i + loss_j
        else:
            attr_i = self.attrregressor_net(feature)
            loss_attr = self.criterion_attr(attr_i, attr_data)

        loss_attr= loss_attr * label
        if label.sum() > 0:
            # no supervised attr in batch
            loss_attr = loss_attr.sum()/label.sum()/attr_data.size(1)
            self.loss += loss_attr
            self.loss_dict["attr_loss"] = loss_attr

    def log_results(self, epoch, batch_idx, dataset_i):
        opts = self.opts

        batches_done = (epoch - opts.init_epoch) * self.num_steps_per_epoch + batch_idx
        batches_left = (opts.n_epochs - opts.init_epoch) * self.num_steps_per_epoch - batches_done
        time_left = datetime.timedelta(seconds=batches_left*(time.time() - self.prev_time))
        self.prev_time = time.time()
        message = (
            f"Epoch: {epoch}/{opts.n_epochs}, Dataset: {dataset_i}/{self.num_dataset}, "
            f"Batch: {batch_idx}/{self.num_steps_per_epoch}, ETA: {time_left}, "
            f"loss_total: {self.loss.item():.6f}, "
        )
        self.curr_step = (epoch - 1)*self.num_steps_per_epoch + batch_idx + 1
        wandb_dict = self.loss_dict
        wandb_dict['step'] = self.curr_step
        if opts.train_attr:
            if self.label.sum() > 0:  # no logging when no attr label
                loss_attr = self.loss_dict["attr_loss"]
                message += f"loss_attr: {loss_attr.item():.6f}"
        if opts.use_wandb:
            wandb.log(wandb_dict)
        print(message)
        self.logfile.write(message + '\n')
        self.logfile.flush()


    def infer_model(self, test_dataloader):
        char_num = test_dataloader.dataset.char_num
        feat_per_font = []
        attr_per_font = []
        attrgt_per_font = []
        self.fontemb_net.eval()
        device = next(self.fontemb_net.parameters()).device # NOTE: assume model in one gpu

        do_attr_extraction = (self.attrregressor_net and test_dataloader.dataset.use_attr)

        with torch.no_grad():
            ## load data for each font with batchsize=charnum
            for ii, batch_test in enumerate(test_dataloader):
                img_i = batch_test['img_i'].to(device)
                feat_i, _  = self.fontemb_net(img_i)
                feat_per_font.append(feat_i)
                if do_attr_extraction:
                    attr_gt = batch_test['attr'].to(device)

                    attr_i = self.attrregressor_net(feat_i)
                    attr_i = self.attr_logit(attr_i)
                    attr_per_font.append(attr_i)
                    attrgt_per_font.append(attr_gt)
                else:
                    attr_i = None
                    attr_gt = None

        feat_per_font = torch.stack(feat_per_font) ##  torch.Size([28, 52, 512])
        feat_per_char = feat_per_font.permute(1,0,2) #  torch.Size([52, 28, 512])
        if do_attr_extraction:
            attr_per_font = torch.stack(attr_per_font) ##  torch.Size([28, 52, 512])
            attr_per_char = attr_per_font.permute(1,0,2) #  torch.Size([52, 28, 512])
            attr_gt = torch.stack(attrgt_per_font)[:,0,:] ## torch.Size([28, 37]) ## donovan
        else:
            attr_per_char = []
                
        print("Extract feat done: {}".format(len(feat_per_char)))
        return feat_per_char, attr_per_char, attr_gt



    def evaluate(self, epoch, test_dataloader):
        opts = self.opts
        self.fontemb_net.eval()
        if self.fontdec_net:
            self.fontdec_net.eval()
        fontemb_net = self.fontemb_net
        attrregressor_net = self.attrregressor_net

        with torch.no_grad():
            """
            TEST font retrieval
            use init_char to test all, or capitals
            """
            t = time.time()
            feat_per_char, attr_per_char, attr_gt = self.infer_model(test_dataloader)
            ret_accuracy, ret_per_char = retrieval_evaluation(feat_per_char, test_dataloader, init_char=0)
            # ret_accuracy_upper, ret_per_char_upper = retrieval_evaluation(feat_per_char, test_dataloader, init_char=26)

            if ret_accuracy > self.best_ret_accuracy:
                self.best_ret_accuracy = ret_accuracy 
                if opts.use_wandb:
                    wandb.log({
                        "best retrieval accuracy" : self.best_ret_accuracy,
                        "best epoch" : epoch,
                        })
            if opts.use_wandb:
                wandb_dict = {}
                wandb_dict["epoch"] = epoch
                wandb_dict["step"] = self.curr_step
                wandb_dict["retrieval accuracy_all"] = ret_accuracy
                # wandb_dict["retrieval accuracy_upper"] = ret_accuracy_upper
                wandb.log(wandb_dict)
                ret_per_char["epoch"] = epoch
                ret_per_char["step"] = self.curr_step
                wandb.log(ret_per_char)
            message = (
                f"Evaluation on testset took {(time.time() - t)/60:.2f} minutes, "
                f"Retrieval accuracy: {ret_accuracy:.6f}, "
            )
            print(message)
            self.val_logfile.write(message + '\n')
            self.val_logfile.flush()
            """
            L1 generation measure
            """
            if self.opts.train_cae and epoch % self.opts.check_L1_gen_freq == 0 :
                t = time.time()
                L1_error = L1_gen_evaluation(
                        self.fontemb_net, 
                        self.fontdec_net,
                        test_dataloader, True)

                message = (
                    f"Evaluation (L1 generation) on testset took {(time.time() - t)/60:.2f} minutes, "
                    f"L1 generation error: {L1_error:.6f}, "
                )
                print(message)

                if opts.use_wandb:
                    wandb.log({
                        'epoch' : epoch,
                        'L1-image-error_{}'.format(test_dataloader.dataset.dataset_name) : L1_error,
                        })
            """
            Attribute loss
            """
            if opts.train_attr:
                t = time.time()
                attrregressor_net.eval()
                error_attr, attr_error_dict, char_error_dict, message = \
                        attribute_evaluation(attr_per_char, attr_gt,
                                test_dataloader,
                                self.criterion_attr_eval)
                message += (f"Evaluation on testset took {(time.time() - t)/60:.2f} minutes,"
                            f"total attr_error : {error_attr.item():.6f}\n")
                if opts.use_wandb:
                    attr_error_dict["epoch"] = epoch
                    attr_error_dict["step"] = self.curr_step
                    attr_error_dict["attr_error"] = error_attr.item()
                    char_error_dict["epoch"] = epoch
                    char_error_dict["step"] = self.curr_step
                    wandb.log(attr_error_dict)
                    wandb.log(char_error_dict)
                print(message)
                self.val_logfile.write(message + '\n')
                self.val_logfile.flush()
                # train()
                attrregressor_net.train()
        # train()
        fontemb_net.train()
        if self.fontdec_net:
            self.fontdec_net.train()

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_epoch", type=int, default=1, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
    parser.add_argument("--multi_gpu", action="store_true", help="whether or not multi gpus")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    # Data
    parser.add_argument("--data_root", type=str, default="data", help="path to data root")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="name of the dataset")
    parser.add_argument("--data_type", type=str, default='2glyphs',
            choices=['1glyph', '2glyphs', 'glyph-set'], help='data type. How many glyps to be sampled.')
    parser.add_argument("--img_size", type=int, default=64, help="image size")
    parser.add_argument("--unsuper_num", type=int, default=968, help="donovan unsupervised samples")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--n_threads", type=int, default=32, help="number of threads of dataloader")
    parser.add_argument("--n_style", type=int, default=4, help="number of style input images")
    parser.add_argument("--no_augmentation", action="store_true", help="disable augmentation")
    parser.add_argument("--identical_augmentation", action="store_true", help="perform identical augmentation on two-glyphs")
    # Channel
    parser.add_argument("--channel", type=int, default=3, help="image channel")
    parser.add_argument("--attr_channel", type=int, default=37, help="attributes channel")
    parser.add_argument("--attr_embed", type=int, default=64,
                        help="attribute embedding channel, attribute id to attr_embed, must same as image size")
    parser.add_argument("--style_out_channel", type=int, default=128, help="number of style embedding channel")
    parser.add_argument("--attr_zero", action="store_true", help="zero out all attributes")
    parser.add_argument("--style_zero", action="store_true", help="zero out all styles")
    parser.add_argument("--n_res_blocks", type=int, default=16, help="number of residual blocks in style encoder")
    # Model
    parser.add_argument("--attention", type=bool, default=True, help="whether use the self attention layer in the generator")
    parser.add_argument("--dis_pred", type=bool, default=True, help="whether the discriminator predict the attributes")
    parser.add_argument("--exterial_style_enc_down", action="store_true", help="use --backbone as style_enc.down network")
    parser.add_argument("--load_style_enc", type=str, default='', help="path to load style_enc.down")
    parser.add_argument("--cut_grad_styl_enc_down", action="store_true", help="whether the discriminator predict the attributes")
    # Junho
    parser.add_argument("--backbone", type=str, default='ResNet18', 
            choices=['ResNet34', 'ResNet18'], help='Choose backbone')
    parser.add_argument("--pretrained", type=str, default='', help="pth path to train from pretrained")
    parser.add_argument('--heads', nargs='+', help='Dim of heads (projection or pui) in list. if None, no heads.', default=None)
    parser.add_argument("--norm_method", type=str, default="BN", help="[BN or IN] for norm")
    parser.add_argument("--train_attr", action="store_true", help="Train Attributes")
    parser.add_argument("--train_fontcls", action="store_true", help="Train font classification")
    parser.add_argument("--train_charcls", action="store_true", help="Train char classification")
    parser.add_argument("--train_ae", action="store_true", help="Train glyph autoencoder")
    parser.add_argument("--train_cae", action="store_true", help="Train glyph conditional-autoencoder")
    parser.add_argument("--feat_dim", type=int, default=512, help="dimension of feature")
    parser.add_argument("--simclr", action="store_true", help="Train Attributes")
    parser.add_argument("--supcon", action="store_true", help="Train Attributes")
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature of nt_xent")
    parser.add_argument("--grad_cut_fontdec", action="store_true", help="If true, do not backprop pixel gradient to fontemb")
    # Adam
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    # parser.add_argument("--lr_fontembcls", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    # Experiment
    parser.add_argument("--experiment_name", type=str, default="att2font_en", help='experiment name')
    parser.add_argument("--check_freq", type=int, default=10, help='frequency of checkpoint epoch')
    parser.add_argument("--check_L1_gen_freq", type=int, default=500, help='frequency of checkpoint epoch')
    parser.add_argument("--sample_freq", type=int, default=400, help="frequency of sample validation batch")
    parser.add_argument("--log_freq", type=int, default=100, help="frequency of sample training batch")
    parser.add_argument("--phase", type=str, default='train',
            choices=['train', 'test', 'test_interp', 'train-representation', 'test-representation'], help='mode')
    parser.add_argument("--test_epoch", type=int, default=0, help='epoch to test, 0 to test all epoches')
    parser.add_argument("--interp_cnt", type=int, default=11, help='number of interpolations')
    # Test
    parser.add_argument("--test_num", type=int, default=100, help="num of testset")
    # Lambdas
    parser.add_argument("--lambda_fontcls", type=float, default=1.0, help='font class loss lambda')
    parser.add_argument("--lambda_l1", type=float, default=50.0, help='pixel l1 loss lambda')
    parser.add_argument("--lambda_char", type=float, default=3.0, help='char class loss lambda')
    parser.add_argument("--lambda_GAN", type=float, default=5.0, help='GAN loss lambda')
    parser.add_argument("--lambda_cx", type=float, default=6.0, help='Contextual loss lambda')
    parser.add_argument("--lambda_attr", type=float, default=20.0, help='discriminator predict attribute loss lambda')
    # Other Modules
    return parser

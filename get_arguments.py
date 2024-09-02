import argparse
import torchvision.models as models
import utils_dino as utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_arguments():
    parser = argparse.ArgumentParser(description='SurgeNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-batch-size', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--disc-learning-rate', default=0.0002, type=float, metavar='LR',
                        help=' learning rate for discriminator')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.2, type=float,
                        help='softmax temperature (default: 0.2)')

    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='path to checkpoint directory')
    parser.add_argument('--train_list', default='dataset/Xray14_train_official.txt', type=str,
                         help='file for training list')
    parser.add_argument('--val_list', default='dataset/Xray14_val_official.txt', type=str,
                         help='file for validation list')
    parser.add_argument('--mode', default='dira', type=str,
                         help='di|dir|dira')
    parser.add_argument('--encoder_weights', default=None, type=str,help='encoder pre-trained weights if available')
    parser.add_argument('--activate', default="sigmoid", type=str,help='activation for reconstruction')
    parser.add_argument('--contrastive_weight', default=1, type=float,help='weight of instance discrimination loss')
    parser.add_argument('--mse_weight', default=10, type=float,help='weight of reconstruction loss')
    parser.add_argument('--adv_weight', default=0.001, type=float,help='weight of adversarial loss')
    parser.add_argument('--exp_name', default="DiRA_moco", type=str,help='experiment name')
    parser.add_argument('--out_channels', default=1, type=str,help='number of channels in generator output')
    parser.add_argument('--generator_pre_trained_weights', default=None, type=str,help='generator pre-trained weights')


    ### extra arguments
    # New argument for experiment name
    parser.add_argument("--name",
        "--experimentname",
        default="default_experiment",
        type=str,
        metavar="NAME",
        help="name of the experiment (default: default_experiment)",
    )

    return parser
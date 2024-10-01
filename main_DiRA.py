import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import transformation
from DiRA_models import DiRA_UNet, DiRA_MoCo, MoCo, Discriminator, weights_init_normal
from trainer import train_dir, validate_dir, train_dira, validate_dira
from torch.autograd import Variable
import utils_dino as utils
import data_loader
from get_arguments import get_arguments


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('caformer_s18')

def train_dira(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(7)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.world_size)

    # ============ preparing data ... ============
    print('===> Preparing data ...')
    dataset = data_loader.concat_zip_datasets(train_dir, transforms=transformation.Transform(args.mode))
    # split dataset in train en validation
    dataset_train = torch.utils.data.Subset(dataset, range(0, int(len(dataset)*0.95)))
    dataset_valid = torch.utils.data.Subset(dataset, range(int(len(dataset)*0.95), len(dataset)))

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=data_loader.custom_collate_fn
    )
    sampler_valid = torch.utils.data.DistributedSampler(dataset_valid, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=data_loader.custom_collate_fn
    )

    print("Train dataset size: ", len(dataset_train))
    print("Validation dataset size: ", len(dataset_valid))

    #  first check if base encoder is in metaformer file
    if args.arch in metaformer.__dict__.keys():
        print("Using metaformer architecture")
        base_encoder = metaformer.__dict__[args.arch](num_classes=args.moco_dim)
    else:
        base_encoder = models.__dict__[args.arch](pretrained=True)

    if args.mode.lower() == "di": #discriminator only
        model = MoCo(base_encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    else:
        model = DiRA_MoCo(base_encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)
    discriminator = Discriminator(args.out_channels)
    discriminator.apply(weights_init_normal)
    print(discriminator)

    if args.generator_pre_trained_weights is not None:
        print ("Loading pre-trained weights for generator...")
        ckpt = torch.load(args.generator_pre_trained_weights, map_location='cpu')
        if "state_dict" in ckpt:
            ckpt = ckpt['state_dict']
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        msg = model.load_state_dict(ckpt)
        print("=> loaded pre-trained model '{}'".format(args.generator_pre_trained_weights))
        print("missing keys:", msg.missing_keys)

        model.cuda()
        discriminator.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])

    nce_criterion = nn.CrossEntropyLoss().cuda()
    mse_criterion = nn.MSELoss().cuda()
    adversarial_criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_D = torch.optim.Adam(
                    params=discriminator.parameters(),
                    lr=args.disc_learning_rate,
                    betas=[0.5, 0.999]) #set from inpainting paper)


    best_loss = 10000000000

    ## wandb logging
    if utils.is_main_process():
        wandb.init(project='DiRA', name=args.name)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.mode.lower() == "di" or args.mode.lower() == "dir":
            train_dir(train_loader, model, nce_criterion, mse_criterion, optimizer, epoch, args)
            counter = validate_dir(valid_loader, model, nce_criterion, mse_criterion, epoch, args)
        elif args.mode.lower() =="dira":
            train_dira(train_loader, model, nce_criterion, mse_criterion, adversarial_criterion, optimizer, epoch,args, discriminator, optimizer_D, D_output_shape)
            counter = validate_dira(valid_loader, model, nce_criterion, mse_criterion, adversarial_criterion, epoch,args, discriminator, D_output_shape)


        if utils.is_main_process():
            valid_loss = counter[0]/counter[1]
            wandb.log({"valid_loss": valid_loss})
            print ("validation loss: ",valid_loss)
            if valid_loss < best_loss:
                print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}".format(epoch, best_loss, valid_loss))
                best_loss = valid_loss
                if args.mode.lower() == "di":
                    torch.save(model.module.encoder_q.state_dict(), os.path.join(args.checkpoint_dir, 'best_checkpoint.pth'))
                else:
                    torch.save(model.module.encoder_q.backbone.state_dict(), os.path.join(args.checkpoint_dir,'best_checkpoint.pth'))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_loss':best_loss
            }, os.path.join(args.checkpoint_dir,'checkpoint.pth'))

            if args.mode.lower() == "dira":
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': discriminator.state_dict(),
                    'optimizer' : optimizer_D.state_dict(),
                }, os.path.join(args.checkpoint_dir,'D_checkpoint.pth'))


    if utils.is_main_process():
        if args.mode.lower() == "di":
            torch.save(model.module.encoder_q.state_dict(), os.path.join(args.checkpoint_dir, 'caformer.pth'))
        else:
            torch.save(model.module.encoder_q.backbone.state_dict(),
                   os.path.join(args.checkpoint_dir,'unet.pth'))


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()

    # Modify the output directory to include the experiment name
    opt.output_dir = os.path.join(opt.output_dir, opt.experiment)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    main()

import argparse
import os
import sys
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from utils import *


###################################################################
# TO DO: set path to ImageNet
# This directory needs to have 'train' and 'val' subdirectories.
IMAGENET_PATH = '/mnt/data/ImageNet/ILSVRC2012/'
###################################################################


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
### Blur-Training additional arguments
parser.add_argument('--exp-name', '-n', type=str, default='',
                    help='Experiment name.')
parser.add_argument('--mode', type=str,
                    choices=['normal', 'all', 'mix',
                             'single-step', 'reversed-single-step',
                             'multi-steps', 'multi-steps-cbt'],
                    help='Training mode.')
parser.add_argument('--blur-val', action='store_true', default=False,
                    help='Blur validation data.')
parser.add_argument('--kernel-size', '-k', type=int, nargs=2, default=(3, 3),
                    help='Kernel size of Gaussian Blur.')
parser.add_argument('--sigma', '-s', type=float, default=1,
                    help='Sigma of Gaussian Blur.')
parser.add_argument('--init-sigma', type=float, default=2,
                    help='Initial Sigma of Gaussian Blur. (multi-steps-cbt)')
parser.add_argument('--cbt-rate', type=float, default=0.9,
                    help='Blur decay rate (multi-steps-cbt)')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # directories settings
    os.makedirs('../logs/outputs', exist_ok=True)
    os.makedirs('../logs/models/{}'.format(args.exp_name), exist_ok=True)
    os.makedirs('../logs/tb', exist_ok=True)

    outputs_path = '../logs/outputs/{}.log'.format(args.exp_name)
    if os.path.exists(outputs_path):
        print('ERROR: This experiment name is already used. \
                Use another name for this experiment by \'--exp-name\'')
        sys.exit()
    # recording outputs
    sys.stdout = open(outputs_path, 'w')
    sys.stderr = open(outputs_path, 'a')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        print('random seed: {}'.format(args.seed))

    # call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # print settings
    print_settings(model, args)

    cudnn.benchmark = True  # for fast run

    # Data loading code
    traindir = os.path.join(IMAGENET_PATH, 'train')
    valdir = os.path.join(IMAGENET_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # run only valitation
        validate(val_loader, model, criterion, args)
        return

    # recording settings
    models_path = '../logs/models/{}/'.format(args.exp_name)
    tb_path = '../logs/tb/{}'.format(args.exp_name)  # TB: tensorboard
    # tensorboardX
    writer = SummaryWriter(log_dir=tb_path)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)  # decay by 10 every 20 epoch

        # train for one epoch
        train_time = time.time()
        loss, acc1, acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
        mins = (time.time() - train_time) / 60
        print("Training time: {:.4f}mins".format(mins))
        # record the values in tensorboard
        writer.add_scalar('loss/train', loss, epoch + 1)  # average loss
        writer.add_scalar('acc1/train', acc1, epoch + 1)  # average acc@1
        writer.add_scalar('acc5/train', acc5, epoch + 1)  # average acc@5

        # evaluate on validation set
        val_time = time.time()
        loss, acc1, acc5 = validate(val_loader, model, criterion, args)
        mins = (time.time() - val_time) / 60
        print("Validation time: {:.4f}mins".format(mins))
        # record the values in tensorboard
        writer.add_scalar('loss/val', loss, epoch + 1)  # average loss
        writer.add_scalar('acc1/val', acc1, epoch + 1)  # average acc@1
        writer.add_scalar('acc5/val', acc5, epoch + 1)  # average acc@5

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, models_path, epoch + 1)
        if (epoch + 1) % 10 == 0:  # save model every 10 epochs
            save_model({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, models_path, epoch + 1)

    writer.close()  # close tensorboardX writer


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1))

    # switch to train mode
    model.train()

    # blur settings
    if args.mode == 'normal':
        args.sigma = 0  # no blur
    elif args.mode == 'multi-steps-cbt':
        args.sigma = adjust_multi_steps_cbt(args.init_sigma, epoch, args.cbt_rate, every=5)  # sigma decay every 5 epoch
    elif args.mode == 'multi-steps':
        args.sigma = adjust_multi_steps(epoch)
    elif args.mode == 'single-step':
        if epoch >= args.epochs // 2:
            args.sigma = 0
    elif args.mode == 'reversed-single-step':
        if epoch < args.epochs // 2:
            args.sigma = 0

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # blur images
        if args.sigma != 0:  # skip if sigma = 0 (no blur)
            if args.mode == 'mix':
                half1, half2 = images.chunk(2)
                # blur first half images
                half1 = GaussianBlurAll(half1, (0, 0), args.sigma)
                images = torch.cat((half1, half2))
            else:
                images = GaussianBlurAll(images, (0, 0), args.sigma)

        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute outputs
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # blur images
            if args.blur_val:
                images = GaussianBlurAll(images, (0, 0), args.sigma)
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    run_time = time.time()
    main()
    mins = (time.time() - run_time) / 60
    hours = mins / 60
    print("Total run time: {:.4f}mins, {:.4f}hours".format(mins, hours))

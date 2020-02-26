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
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

from utils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
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
### BlurNet additional arguments
parser.add_argument('--exp-name', '-n', type=str, default='',
                    help='Experiment name.')
parser.add_argument('--mode', type=str, choices=['normal', 'blur-all', 'blur-half', 'blur-step', 'blur-half-data'],
                    help='Training mode.')
parser.add_argument('--kernel-size', '-k', type=int, nargs=2, default=(3,3),
                    help='Kernel size of Gaussian Blur.')
parser.add_argument('--sigma', '-s', type=float, default=1,
                    help='Sigma of Gaussian Blur.')

best_acc1 = 0


def main():
    args = parser.parse_args()
    
    # directories settings
    os.makedirs('../logs/outputs', exist_ok=True)
    os.makedirs('../logs/models/{}'.format(args.exp_name), exist_ok=True)
    os.makedirs('../logs/tb', exist_ok=True)
    
    OUTPUT_PATH = '../logs/outputs/{}.log'.format(args.exp_name)
    if os.path.exists(OUTPUT_PATH):
        print('ERROR: This experiment name is already used. Use another name for this experiment by \'--exp-name\'')
        sys.exit()
    # recording outputs
    sys.stdout = open(OUTPUT_PATH, 'w')
    sys.stderr = open(OUTPUT_PATH, 'a')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        print('seed: {}'.format(args.seed))

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    # print settings
    print('='*5 + ' settings ' + '='*5)
    print('TRAINING MODE: {}'.format(args.mode))
    if args.mode == 'blur-step':
        print('### BLUR CHANGING STEPS ###')
        print('Step: 1-10 -> 11-20 -> 21-30 -> 31-40 -> 41-{}'.format(args.epochs))
        print('Sigma: 4 -> 3 -> 2 -> 1 -> none')
        print('Kernel-size: (25, 25) -> (19, 19) -> (13, 13) -> (7, 7) -> none')
        print('#'*20)
    elif args.mode == 'blur-half':
        print('### NO BLUR FROM EPOCH {:d} ###'.format(args.epochs // 2))
        print('Sigma: {}'.format(args.sigma))
        print('Kernel-size: {}'.format(tuple(args.kernel_size)))  # radius = sigma * 3 * 2 + 1
    elif args.mode == 'blur-all':
        print('Sigma: {}'.format(args.sigma))
        print('Kernel-size: {}'.format(tuple(args.kernel_size)))  # radius = sigma * 3 * 2 + 1
    print('Random seed: {}'.format(args.seed))
    print('Epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print('Weight_decay: {}'.format(args.weight_decay))
    print()
    print(model)
    print('='*20)
    print()
    
    torch.backends.cudnn.benchmark=True  # for fast training

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

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
    else:
        print('initial learning: {}'.format(args.lr))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print('batch size: {}'.format(args.batch_size))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    # recording settings
    MODEL_PATH = '../logs/models/{}/'.format(args.exp_name)
    TB_PATH = '../logs/tb/{}'.format(args.exp_name)
    # tensorboardX
    writer = SummaryWriter(log_dir=TB_PATH)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_time = time.time()
        loss, acc1, acc5 = train(train_loader, model, criterion, optimizer, epoch, args)
        mins = (time.time() - train_time) / 60
        print("Training time: {:.4f}mins".format(mins))
        # record the values in tensorboard
        writer.add_scalar('loss/train', loss, epoch + 1)  # avarage loss
        writer.add_scalar('acc1/train', acc1, epoch + 1)  # avarage acc@1
        writer.add_scalar('acc5/train', acc5, epoch + 1)  # avarage acc@5

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

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, MODEL_PATH, epoch + 1)
            if (epoch + 1) % 10 == 0:  # save model every 10 epochs
                save_model({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                    }, MODEL_PATH, epoch + 1)                
            
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
    blur = True
    if args.mode == 'normal':
            blur = False
        elif args.mode == 'blur-step':
            ### BLUR-STEP SETTINGS ###
            if epoch < 10:
                args.sigma = 4
                args.kernel_size = (25, 25)  # radius = sigma * 3 * 2 + 1
            elif epoch < 20:
                args.sigma = 3
                args.kernel_size = (19, 19)
            elif epoch < 30:
                args.sigma = 2
                args.kernel_size = (13, 13)
            elif epoch < 40:
                args.sigma = 1
                args.kernel_size = (7, 7)
            else:
                blur = False
        elif args.mode == 'blur-half':
            if epoch >= args.epochs // 2:
                blur = False

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # blur images
        if blur:
            if args.mode == 'blur-half-data':
                    half1, half2 = images.chunk(2)
                    # blur first half images
                    half1 = GaussianBlurAll(half1, tuple(args.kernel_size), args.sigma)
                    images = torch.cat((half1, half2))
            else:
                images = GaussianBlurAll(images, tuple(args.kernel_size), args.sigma)  
            
        if args.gpu is not None:
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
            # images = GaussianBlurAll(images, tuple(args.kernel_size), args.sigma)
            if args.gpu is not None:
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

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

        

if __name__ == '__main__':
    run_time = time.time()
    main()
    mins = (time.time() - run_time) / 60
    hours = mins / 60
    print("Total run time: {:.4f}mins, {:.4f}hours".format(mins, hours)) 
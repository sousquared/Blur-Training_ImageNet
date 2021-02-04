import random
import shutil

import cv2
import numpy as np
import torch
from torchvision import models


def GaussianBlurAll(imgs, sigma, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    if sigma == 0:
        return imgs  # do nothing
    else:
        imgs = imgs.numpy()
        imgs_list = []
        for img in imgs:
            imgs_list.append(
                cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma)
            )
        imgs_list = np.array(imgs_list)
        imgs_list = imgs_list.transpose(0, 3, 1, 2)
        return torch.from_numpy(imgs_list)


def RandomGaussianBlurAll(
    imgs, min_sigma, max_sigma, kernel_size=(0, 0)
) -> torch.Tensor:
    """Return Blurred images by random sigma.
    Each image is blurred by a sigma chosen by randomly from [min_sigma, max_sigma].

    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    imgs = imgs.numpy()

    imgs_list = []
    for img in imgs:
        # Choose a random sigma for each image
        sigma = random.uniform(min_sigma, max_sigma)

        imgs_list.append(cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma))

    imgs_list = np.array(imgs_list)
    imgs_list = imgs_list.transpose(0, 3, 1, 2)
    return torch.from_numpy(imgs_list)


def adjust_multi_steps(epoch: int):
    """For 'multi-steps' mode.
        Return sigma based on current epoch of training.
    Args:
        epoch: current epoch of training
    Returns: sigma
    """
    if epoch < 10:
        sigma = 4
    elif epoch < 20:
        sigma = 3
    elif epoch < 30:
        sigma = 2
    elif epoch < 40:
        sigma = 1
    else:
        sigma = 0  # no blur

    return sigma


def adjust_multi_steps_cbt(init_sigma, epoch, decay_rate=0.9, every=5):
    """
    Sets the sigma of Gaussian Blur decayed every 5 epoch.
    This is for 'multi-steps-cbt' mode.
    This idea is based on "Curriculum By Texture"
    Args:
        init_sigma: initial sigma of Gaussian kernel
        epoch: training epoch at the moment
        decay_rate: how much the model decreases the sigma value
        every: the number of epochs the model decrease sigma value
    Return: sigma of Gaussian kernel (GaussianBlur)
    """
    return init_sigma * (decay_rate ** (epoch // every))


def adjust_learning_rate(optimizer, epoch, args, every=20):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, param_path, epoch):
    filename = param_path + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, param_path + "model_best.pth.tar")


def save_model(state, param_path, epoch):
    filename = param_path + "epoch_{}.pth.tar".format(epoch)
    torch.save(state, filename)


def load_model(model_path, arch="alexnet"):
    """Load model.
    Args:
        model_path: Path to the pytorch saved file of the model you want to use
        arch: Architecture of CNN
    Returns: CNN model
    """
    checkpoint = torch.load(model_path, map_location="cuda:0")
    model = models.__dict__[arch]()
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        model.features = torch.nn.DataParallel(model.features)
        model.load_state_dict(checkpoint["state_dict"])
        # model.features = model.features.module  # if you want to disable Dataparallel
    return model


def print_settings(model, args):
    print("=" * 5 + " settings " + "=" * 5)
    print("TRAINING MODE: {}".format(args.mode.capitalize()))
    if args.mode == "mix":
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "single-step":
        print("## NO BLUR FROM EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "reversed-single-step":
        print("## NO BLUR TILL EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "multi-steps":
        print("Step: 1-10 -> 11-20 -> 21-30 -> 31-40 -> 41-{}".format(args.epochs))
        print("Sigma: 4 -> 3 -> 2 -> 1 -> none")
        print("#" * 20)
    elif args.mode == "all":
        print("Sigma: {}".format(args.sigma))
    if args.blur_val:
        print("VALIDATION MODE: blur-val")
    print("Batch size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print(
        "Initial Learning rate: {} (It will be decayed by 10 every 20 epochs)".format(
            args.lr
        )
    )
    print("Weight_decay: {}".format(args.weight_decay))
    print()
    print(model)
    print("=" * 20)

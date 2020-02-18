import sys 
sys.path.append('../')
import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .utils import dataloader, GaussianBlur_images
from .models import Net, AlexNetCifar10

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--exp-name', '-n', type=str, default='',
                    help='Experiment name.')
parser.add_argument('--normal', action='store_true', default=False,
                    help='Normal training mode (w/o blurring images).')
parser.add_argument('--kernel-size', '-k', type=int, nargs=2, default=(3,3),
                    help='Kernel size of Gaussian Blur.')
parser.add_argument('--sigma', '-s', type=float, default=1,
                    help='Sigma of Gaussian Blur.')
parser.add_argument('--epochs', '-e', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', '-b', type=int, default=64,
                    help='Batch size.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight-decay', '-w', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()
if args.exp_name == '':
    print('ERROR: USE \'--exp-name\' or \'-n\' option to define this experiment\'s name.')
    sys.exit()

# directories settings
os.makedirs('../logs/results', exist_ok=True)
os.makedirs('../logs/params', exist_ok=True)

RESULT_PATH = '../logs/results/{}.txt'.format(args.exp_name)
MODEL_PATH = '../logs/params/{}.pth'.format(args.exp_name)


if os.path.exists(RESULT_PATH):
    print('ERROR: This \'--exp-name\' is already used. Use another name for this experiment.')
    sys.exit()

# recording outputs
sys.stdout = open(RESULT_PATH, 'w')
sys.stderr = open(RESULT_PATH, 'a')

# tensorboardX
writer = SummaryWriter(log_dir='../logs/tb/{}'.format(args.exp_name))
 
# cuda settings
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
print('device: {}'.format(device))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# data settings
trainloader, testloader, classes = dataloader(batch_size=args.batch_size)

# Model, Criterion, Optimizer
net = AlexNetCifar10(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

# print settings
print('='*5 + ' settings ' + '='*5)
print('TRAINING MODE: {}'.format('Blur' if not args.normal else 'Normal'))
if not args.normal:
    print('Kernel-size: {}'.format(tuple(args.kernel_size)))
    print('Sigma: {}'.format(args.sigma))
print('Random seed: {}'.format(args.seed))
print('Epochs: {}'.format(args.epochs))
print('Learning rater: {}'.format(args.lr))
print('Weight_decay: {}'.format(args.weight_decay))
print()
print(net)
print('='*20)
print()

# training
num_trainbatch = -(-len(trainloader.dataset) / trainloader.batch_size) # e.g. 50000 / 4 (cifar10)
num_testbatch = -(-len(testloader.dataset) / testloader.batch_size)  # e.g. 10000 / 4 (cifar10)
print('Start Training...')
train_time = time.time()
for epoch in range(args.epochs):  # loop over the dataset multiple times
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    # ===== train mode =====
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Blur images
        if not args.normal:
            inputs = GaussianBlur_images(inputs.cpu(), \
                                         tuple(args.kernel_size), args.sigma)  
            inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()        
        loss.backward()
        optimizer.step()
    
    # record the values in tensorboard
    writer.add_scalar('loss/train', train_loss / num_trainbatch, epoch + 1)  # avarage loss
    writer.add_scalar('acc/train', train_acc / len(trainloader.dataset), epoch + 1)  # avarage acc
        
    # ===== val mode =====
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            """
            if not args.normal:
                inputs = GaussianBlur_images(inputs.cpu(), \
                                             tuple(args.kernel_size), args.sigma) 
                inputs = inputs.to(device)
            """
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    
    # record the values in tensorboard
    writer.add_scalar('loss/val', val_loss / num_testbatch, epoch + 1)  # avarage loss
    writer.add_scalar('acc/val', val_acc / len(testloader.dataset), epoch + 1)  # avarage acc

print('Finished Training')
print("Training time elapsed: {:.4f}mins".format((time.time() - train_time) / 60))
print()

# save the model
torch.save(net.state_dict(), MODEL_PATH)


# test
net = AlexNetCifar10(num_classes).to(device)
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if not args.normal:
    print('NOTE: Test images are NOT blurred.')
    #print('NOTE: Test images are blurred.')
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


writer.close()  # close tensorboardX writer

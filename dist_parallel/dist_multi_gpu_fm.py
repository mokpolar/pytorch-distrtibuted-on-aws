import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import argparse
from tensorboardX import SummaryWriter


from torchvision.models.resnet import ResNet, BasicBlock



parser = argparse.ArgumentParser(description='Fashion MNIST classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_worker', type=int, default=2, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://52.23.194.120:8889', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='') # backend. in this case I use nccl. 
parser.add_argument('--rank', default=0, type=int, help='') # numbers of node. I will Use 4 terminals. node0 : 0, 1, node1 : 2, 3
parser.add_argument('--world_size', default=2, type=int, help='') # use 1 when I use local only
parser.add_argument('--distributed', action='store_true', help='') 
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


def main():

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))



def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print("Initialize Model...")

    # Construct Model

    net = MnistResNet()
    torch.cuda.set_device(args.gpu)
    net.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    
    # define loss function (criterion) and optimizer 
    criterion = nn.CrossEntropyLoss().cuda() 
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    print("Initialize Dataloaders...")

    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = datasets.FashionMNIST(root = './data', download=True, train=True, transform=transform)
    valset = datasets.FashionMNIST(root = './data', download=True, train=False, transform=transform)

    # need sampler for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)


    # Fashion MNIST Dataloader
    train_loader = DataLoader(trainset, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), num_workers=args.num_workers, 
                              sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_worker, pin_memory = False)
    



    # device는 cuda로.
    train(net, criterion, optimizer, train_loader, args.gpu) 
            

def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total
        
        batch_time = time.time() - start
        
        if batch_idx % 20 == 0:
            print('Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s '.format(
                batch_idx, len(train_loader), train_loss/(batch_idx+1), acc, batch_time))
    
    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))
    

if __name__=='__main__':
    main()
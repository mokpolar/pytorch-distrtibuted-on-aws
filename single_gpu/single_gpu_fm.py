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

import argparse
from tensorboardX import SummaryWriter


from torchvision.models.resnet import ResNet, BasicBlock



parser = argparse.ArgumentParser(description='Fashion MNIST classification models')
parser.add_argument('--lr', default=0.1, help='') # -> 여기에 learning rate decay를 적용해야 함. 이것도 비교에 영향을 줄테니까. 
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--num_worker', type=int, default=2, help='') # 동일 선상의 비교를 위해 batch_size와 num_workers는 동일 지정.
args = parser.parse_args()



class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), bias=False)


def main():
    best_acc = 0 # best accuracy

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 디바이스는 쿠다를 사용한다 쿠다거 어베일러블 한다면. else인 경우는 cpu를 사용. 


    print("Initialize Dataloaders...")

    transform = transforms.Compose(
        [transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  rgb scale -> gray scale

    # Download and load the training data
    trainset = datasets.FashionMNIST(root = './data', download=True, train=True, transform=transform)
    valset = datasets.FashionMNIST(root = './data', download=True, train=False, transform=transform)

    # Fashion MNIST Dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_worker, pin_memory = False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_worker, pin_memory = False)
 

    print("Initialize Model...")

    # Construct Model
    #model = models.resnet18(pretrained=False)#.cuda()
    #model = MnistResNet()
    net = MnistResNet() # 이거네. PYRAMIDNET 구조 갖고오고. 이건 같은 폴더에 model.py에서 갖고옴. 
    net = net.to(device) # 이걸 디바이스에 넣나봄. CUDA에? 
    # define loss function (criterion) and optimizer 
    criterion = nn.CrossEntropyLoss()#.cuda() # 이 cuda를 쓴다는 건 GPU Tensor로 만든다는 건데. 
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

    print('==> Making model..')

    # device는 cuda로.
    train(net, criterion, optimizer, train_loader, device) # 그리고 TRAIN METHOD 이건 아래에 TRAIN METHOD 별도 선언한 것. 
            

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
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
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from model import pyramidnet
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_worker', type=int, default=4, help='') # 이건 CPU 갯수인가? 그런 것 같아.
args = parser.parse_args() # args 는 이것들을 아마 출력 가능한 형태로 뽑는 것 같아. 


def main():
    best_acc = 0 # 뭔지 모르겠다만

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # 디바이스는 쿠다를 사용한다 쿠다거 어베일러블 한다면. else인 경우는 cpu를 사용. 

    print('==> Preparing data..') # 데이터 준비는 TORCH VISION의 TRANSFROM 라이브러리를 사용한다. PYTORCH의 데이터 전처리 패키지라고 한다. 
    transforms_train = transforms.Compose([ # 아닌가? transforms_TRAIN 은 COMPOSE METHOD를 사용하는데, 
        transforms.RandomCrop(32, padding=4), # 여러 ARGS를 지정하는데. 그냥 데이터 섞는 방식인거 같은데?
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    dataset_train = CIFAR10(root='../data', train=True, download=True,  # 진짜 데이터는 여기 있다. CIFAR10을 갖고 와서 저장한다. TORCH VISION에 들어있네
                            transform=transforms_train) # TEST만 별도로 갖고 올 수도 있고, 없으면 DOWNLOAD해서 쓰나봄. 

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size,  
                              shuffle=True, num_workers=args.num_worker # 데이터 로더는 데이터셋 트레인 가지고 , 기존에 지정한 batch size와 shuffle을 하고, num workers도 지정.CPU갯수다이건

    # there are 10 classes so the dataset name is cifar-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck') # CIFAR CLASS 지정하고 

    print('==> Making model..')

    net = pyramidnet() # 이거네. PYRAMIDNET 구조 갖고오고. 이건 같은 폴더에 model.py에서 갖고옴. 
    net = net.to(device) # 이걸 디바이스에 넣나봄. CUDA에? 
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad) # NET의 PARAMETER들 하나하나를 뽑아서 파라미터 수의ㅏ 합을 저장하네 이건 왜ㅕ하지 
    print('The number of parameters of model is', num_params) # 아 파라미터 숫자 보여주려고. 이런 식으로 뺄 수 있구만

    criterion = nn.CrossEntropyLoss() # CRITERION은 LOSS FUNCTION을 이렇게 부르는 건가? CROSSENTROPY를 썼다. nn 안에 들어있네. 
    optimizer = optim.SGD(net.parameters(), lr=args.lr,  
                          momentum=0.9, weight_decay=1e-4) # OPTIMIZER는 Stochastic Gradinet Descent를 사용. lr은 기 입력된 대로. monentum. weight_decay도 적용
    
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
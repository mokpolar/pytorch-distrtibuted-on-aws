import time
import sys
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.multiprocessing import Pool, Process

# the default is fork which may cause deadlocks when using multiple worker processes for dataloading.
# 주로 필요한 library  torch.nn.parallel, torch.distributed, torch.utils.data.distributed, and torch.multiprocessing


# 이 부분은 Helper Function을 담당하는 Class
# 해당 클래스는 정확도 및 반복 횟수 같은 Training 관련 statistics를 추적한다. 

class AverageMeter(object): # average와 현재 value를 계산하고 저장하는 클래스이다. 
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()

    def reset(self): # 기본적으로 리셋을 해주고 시작하고. 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # update는 val sum count avg가 새로 생기면 업데이트. 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
# accuracy 함수는 학습 진척도를 추적할 수 있도록 tok-k 정확도 (모델의) 를 계산하고 반환한다. 
# Both are provided for training convenience but neither are distributed training specific.
# 둘 다 훈련 편의를 위해 제공되지만 분산 훈련은 특정되지 않는다. 이게 무슨 말인가. 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k""" ## PRECISION을 계산하는 건데. SPECIFIED된 VALUE, K중에서 를 위해 계산하는거다. 
    with torch.no_grad():  # torc.no_grad()를 열어서 시작한다. NO_GRAD가 뭘까.
    # 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 with torch.no_grad(): 로 감쌀 수 있습니다. 
    # 이는 특히 변화도(gradient)는 필요없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용합니다.
    # model이 이미 설정되어 있는 것에 사용한다는 거군. 
        maxk = max(topk)
        batch_size = target.size(0) # batchsize는 타겟의 사이즈에 맞춰서 정하는 듯. 

        _, pred = output.topk(maxk, 1, True, True) # 아웃 풋을 쪼개서.? output의 topk가 뭐길래. pred랑 필요없는게 같이 들어있나?
        pred = pred.t() # pred를 전치시켜서 남김. 형태가 다른가보네. 
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # pred.eq(data)는 pred배열과 data가 일치하느냐를 검사 그 뒤에 .sum()을 붙임으로 인해서 일치하는 것들의 개수의 합을 숫자로 출력
        # 이렇게 맞는게 얼마나 되는지를 체크하는 거구나. 

        res = [] # 비워두고 
        for k in topk:  # TOPK 를 ITERATION 돌리면서. CORRECT에서 K마다. 새로 하나하나 뽑아서
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # correct에서 k를 하나하나 view 비교 하는건가. sum해가지고?  뭐 맞는거 본다고 생각하자 일단. 
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#**메인 루프를 단순화하기 위해서는 TRAINING EPOCH STEP을  TRAIN 이라고 불리는 FUNCTION으로 분리하는 것이 가장 좋다.** 
# 이 함수는 train_loader의 1개 EPOCH에 대한 입력 모델을 훈련한다. 
#  기능에서 유일한 분산된 훈련 아티팩트는 데이터의 비_차단  NON BLOCKING ATTRIBUTES특성 및 전진 통과 전에 라벨 텐서를 True로 설정하는 것이다.***

def train(train_loader, model, criterion, optimizer, epoch): # train_loader를 주목하라. criterion은 loss function이었지 

    batch_time = AverageMeter() # 이거 계속 같은걸 가지고 선언하면 어떻게 되는거지 같은 클래스를 다섯 번 복사하는건가. 
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode , model.가지고 train 시작. 
    model.train()

    end = time.time() # 시간을 재자. 
    for i, (input, target) in enumerate(train_loader): # 인덱스를 만들어 가며 input target을 하나씩 꺼낸다. tuple로 . 

        # measure data loading time # data_time은 averagetMeter()클래스로 선언해둔거에 update를 하는건가.  아 통계치를 계속 만드는구나. 이렇게 클래스를 만들어서. 
        data_time.update(time.time() - end) # 현재 시간하고 선언한 시간 차 계산해서 얼마나 걸렸는지를 data_time에 업데이트하며 통계치를 만든다. 

        # Create non_blocking tensors for distributed training 논 블록킹은 바로 반응하는거다. 
        input = input.cuda(non_blocking=True) # nonblocking이란건 간단하게 다른 노드 간 통신이 완료되기 전에 즉각적으로 반응한다는 것인것 같다. 
        target = target.cuda(non_blocking=True) # target.cuda() torch.cuda()는 GPU Tensor이다. GPU연산이 가능한 자료형이라는 뜻. 

        # compute output
        output = model(input) # input을 넣고 output. 
        loss = criterion(output, target) # criterion . loss function.

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5)) # 미리 만들어 둔 accuracy function output과 target을 받아서 topk 1과 5를 계산. 
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad() # gradient를 0으로 만든다. 
        loss.backward() # loss.backward()이건 왜. 

        # Call step of optimizer to update model params
        optimizer.step() # optimizer step? 으로 model parameter 업데이트. 

        # measure elapsed time
        batch_time.update(time.time() - end) # 
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    # 추론을 실행하기 전에는 반드시 model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드로 설정하여야 합니다. 이것을 하지 않으면 추론 결과가 일관성 없게 출력됩니다.

    model.eval()

    with torch.no_grad(): # no grad로 열기. inference만 하니까?
        end = time.time() # 시간 측정
        for i, (input, target) in enumerate(val_loader): # val_loader에서 

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end) # 미리 클래스 선언해둔 batch_time으로 걸린 시간 업데이트. 
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' # top1에 대한 precision. top1.val, avg 그리고 
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg



print("Collect Inputs...")

# Batch Size for training and testing
batch_size = 32

# Number of additional worker processes for dataloading
workers = 2

# Number of epochs to train for
num_epochs = 2

# Starting Learning Rate
starting_lr = 0.1

# Number of distributed processes
world_size = 4 # master node에 전체 node 수가 몇 개인지를 알려줌. 

# Distributed backend type
dist_backend = 'nccl'

# Url used to setup distributed training
# 현재 Instance 기준 Private IP 와 Port
dist_url = "tcp://171.31.35.170:8888"



print("Initialize Process Group...")
# Initialize Process Group
# v1 - init with url
dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
# v2 - init with file
# dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
# v3 - init with environment variables
# dist.init_process_group(backend="nccl", init_method="env://", rank=int(sys.argv[1]), world_size=world_size)


# Establish Local Rank and set device on this node
# local rank는 해당 local PC에서. 노드가 몇번인지.를 말하는 거인듯. 
local_rank = int(sys.argv[2])
dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)


print("Initialize Model...") # 모델 시작
# Construct Model
# 모델은 resnet18 그대로 쓰고 -> model
model = models.resnet18(pretrained=False).cuda() # 이것도 cuda()로 tensor로 만들어줘야 하나? weight는 안 갖고 오고 그냥 구조만 인건가봐. 

# Make model DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)  # device_ids에는 로컬 랭크 지정한거. out_device에는? 로컬랭크 지정한거. 

# define loss function (criterion) and optimizer 
criterion = nn.CrossEntropyLoss().cuda() # loss function은 crossetnropyloss인데 이건 왜 cuda를 해주고 optimizer는 안 해줄까?
optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4) # 이 안에는 왜 model.parameters를 넣어줄까?




print("Initialize Dataloaders...") # 데이터 로더 시작
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
#transform = transforms.Compose( # 여기서 Fashion MNIST에 맞게 변형시켜주면 되는거지. 
#    [transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Compose > Fashion MNIST
# 이렇게 하면 Normalize는 되는데, 위에서 한 기능인 Resize를 적용 못하는거 아닌지. 데이터셋에 따라 다를듯
transform = transforms.Compose( # 여기서 Fashion MNIST에 맞게 변형시켜주면 되는거지. 224로 바꿔줘야 하는 이유가 있나
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Initialize Datasets. STL10 will automatically download if not present
#trainset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
#valset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# STL10 > Fashion MNIST
# Download and load the training data
trainset = datasets.FashionMNIST(root = './data', download=True, train=True, transform=transform)
valset = datasets.FashionMNIST(root = './data', download=True, train=False, transform=transform)


# Create DistributedSampler to handle distributing the dataset across nodes when training
# 이 코드는 노드 별로 데이터셋을 다루기 위한 것이다. 
# This can only be called after torch.distributed.init_process_group is called
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Create the Dataloaders to feed data to the training and validation steps
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
#val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)


# Fashion MNIST Dataloader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers = workers, pin_memory = False, sampler = train_sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers = workers, pin_memory = False)


best_prec1 = 0

for epoch in range(num_epochs):
    # Set epoch count for DistributedSampler
    train_sampler.set_epoch(epoch)

    # Adjust learning rate according to schedule
    adjust_learning_rate(starting_lr, optimizer, epoch)

    # train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(val_loader, model, criterion)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))


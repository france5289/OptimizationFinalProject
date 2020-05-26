# Train ResNet-18 wtih Adam Optimizer
# Dataset : CIFAR-10

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

PATH = './Cifar10_model/Resnet18_Adam.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def SetupSEED():
    '''
    Use this function to setup CUDA and CPU random SEED
    '''
    SEED = 520
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




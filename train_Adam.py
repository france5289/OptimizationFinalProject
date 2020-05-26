# Train ResNet-18 wtih Adam Optimizer
# Dataset : CIFAR-10

import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

PATH = './Cifar10_model/Resnet18_Adam.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def SetupSEED(seed):
    '''
    Use this function to setup CUDA and CPU random SEE
    Args:
    ---
        seed(int) : random seed number
    '''
    torch.cuda.manual_seed(seed)
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

if __name__ == "__main__":
    # Read model config and setup random seed 

    print('Preparing Training and Testing CIFAR-10 dataset')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=16)
    print('Training and Testing dataset is ready...')
    # ==== Load ResNet model ====
    print('Loading ResNet-18 Model')
    resent18 = models.resnet18(pretrained=False)
    print('Deploy to ResNet-18 GPU')
    resent18.to(DEVICE)

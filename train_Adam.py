# Train ResNet-18 wtih Adam Optimizer
# Dataset : CIFAR-10

import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config.Resnet18_config import AdamConfig

CWD = os.getcwd()
HPARAMS_PATH = os.path.join(CWD, 'hyperparameters.json')
PATH = './Cifar10_model/Resnet18_Adam.pth'
TB_PATH = os.path.join(CWD,'exp_log','Adam')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WORKERS = os.cpu_count() // 2


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
    myconfig = AdamConfig.load_from_json(HPARAMS_PATH)
    SetupSEED(myconfig.seed)
    # Download, Read and Pre-process datasets
    print('Preparing Training and Testing CIFAR-10 dataset')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=WORKERS)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=WORKERS)
    print('Training and Testing dataset is ready...')
    # ==== Load ResNet model ====
    print('Loading ResNet-18 Model')
    resent18 = models.resnet18(pretrained=False)
    print('Deploy ResNet-18 to GPU')
    resent18.to(DEVICE)
    # ==== Define Loss function and Adam Optimizer ====
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(resent18.parameters(), lr=myconfig.lrate, weight_decay=myconfig.weight_decay)
    # ==== Setup tensorboard summary writer ====
    if not os.path.exists(TB_PATH):
        os.mkdir(TB_PATH)
    writer = SummaryWriter(os.path.join(TB_PATH, myconfig.expname))
    # ==== Training ====


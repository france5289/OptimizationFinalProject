# Inference ResNet-18 trained with Adam Optimizer
# Dataset : CIFAR-10
import os
import torch
import torchvision
import torchvision.models as models 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

CWD = os.getcwd()
TB_PATH = os.path.join(CWD, 'exp_log')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
WORKERS = os.cpu_count() // 2

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

if __name__ == "__main__":
    # Download, Read and Pre-process test datasets
    print('Preparing Testing CIFAR-10 dataset')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=WORKERS)
    print('Testing dataset is ready...')
    # ==== Load ResNet model state ====
    print('Loading ResNet-18 Model from checkpoint')
    expfname = input('Please input checkpoint filename(ex : test1): ')
    CKPATH = os.path.join(CWD, 'Cifar10_model', expfname + '.pth')
    resnet18 = models.resnet18(pretrained=False)
    resnet18.load_state_dict(torch.load(CKPATH))
    print('Deploy model to GPU')
    resnet18.to(DEVICE)
    # ==== Setup tensorboard summary writer ====
    # if not os.path.exists(TB_PATH):
        # os.mkdir(TB_PATH)
    # writer = SummaryWriter(os.path.join(TB_PATH, expfname))
    # ==== Inference ====
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = resnet18(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images : {(100 * correct / total)}')




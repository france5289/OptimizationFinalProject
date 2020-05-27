# Train ResNet-18 wtih Adam Optimizer
# Dataset : CIFAR-10

import os

import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.Resnet18_config import AdamConfig

CWD = os.getcwd()
HPARAMS_PATH = os.path.join(CWD, 'hyperparameters.json')
TB_PATH = os.path.join(CWD,'exp_log')
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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    # Read model config, setup random seed and model save path
    myconfig = AdamConfig.load_from_json(HPARAMS_PATH)
    SetupSEED(myconfig.seed)
    SAVE_PATH = os.path.join(CWD, 'Cifar10_model', myconfig.expname,'.pth')
    # Download, Read and Pre-process datasets
    print('Preparing Training CIFAR-10 dataset')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=myconfig.batch_size, shuffle=True, num_workers=WORKERS)
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
    loss_list = []
    for epoch in range(myconfig.nepoch):
        print(f'-----Epoch : {epoch}-----')
        loss = 0.0
        mytrange = tqdm(enumerate(trainloader), total=len(trainloader), desc='Train')
        for i,(inputs, labels) in mytrange:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = resent18(inputs)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            # print statistics
            loss += batch_loss.item()
            mytrange.set_postfix(loss=loss / (i+1))
        # ==== Log traning loss of an epoch to tensorboard ====
        epoch_loss = loss/len(mytrange)
        loss_list.append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
    # ==== Save model ====
    torch.save(resent18.state_dict(), SAVE_PATH)
    # ==== Add hyperparameters and min loss to tensorboard ====
    hparams = {
        'seed' : myconfig.seed,
        'lrate' : myconfig.lrate,
        'batch_size' : myconfig.batch_size,
        'epochs' : myconfig.nepoch,
        'weight_decay' : myconfig.weight_decay
    }
    metric = {'min_loss/train':min(loss_list)}
    writer.add_hparams(hparams, metric)
    # ==== Close tensorboard writer ====
    writer.close()


import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import time
import os


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveMaxPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CNNFromPaper(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNFromPaper, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(9, 1), padding=(4, 0)), # Conv1
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=(5, 1), padding=(2, 0)),  # Conv2
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=(3, 1), padding=(1, 0)),  # Conv3
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((62, 64)),  # MaxPool1
            nn.BatchNorm2d(num_features=10),
            nn.Conv2d(10, 40, kernel_size=(3, 1), padding=(1, 0)),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(40, 40, kernel_size=(3, 1), padding=(1, 0)),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # MaxPool2
            nn.BatchNorm2d(num_features=40),
            nn.Conv2d(40, 80, kernel_size=(13, 1), padding=(6, 0)),  # Conv6
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 80, kernel_size=1),  # Conv7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # MaxPool3
            nn.BatchNorm2d(num_features=80),
            nn.Conv2d(80, 80, kernel_size=1),  # Conv8
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=80 * 15 * 64, out_features=80),
            nn.ReLU(inplace=True),
            nn.Linear(80, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'])
        model.load_state_dict(state_dict)
    return model


def train_num_epochs(model, trainloader, testloader, device, criterion, optimizer,
                     starting_epoch, ending_epoch,
                     basic_name='', path_to_weights=''):
    dataset_size = len(trainloader.dataset)
    correct = 0
    total = 0
    best_acc = 0.0
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    # iterate over epochs
    for epoch_num in range(starting_epoch, ending_epoch):
        print('Epoch #%d' % (epoch_num))
        # iterate over batches
        epoch_loss = 0.0
        model.train()
        t0 = time.time()
        for i, (data, target) in enumerate(trainloader):
            print(i)
            data = data.to(device)
            target = target[0]
            target = target.to(device)
            # zero all the gradient tensors
            optimizer.zero_grad()
            # run forward step
            predicted = model(data)
            # compute loss
            loss = criterion(predicted, target)
            # compute gradient tensors
            loss.backward()
            # update parameters
            optimizer.step()
            # compute the training loss value
            epoch_loss += loss.item() * data.size(0)
            total += target.size(0)
            _, pred_labels = torch.max(predicted.data, 1)
            correct += (pred_labels == target).sum().item()
        t = time.time() - t0
        epoch_loss /= dataset_size
        train_acc = correct / total
        print('# Time passed: %.0f s' % (t))
        print('# Epoch loss = %.4f' % (epoch_loss))
        print('# Train acc = {}'.format(train_acc))
        print('# Validation process on validation set')
        val_loss, val_acc = validate(model, criterion, testloader, device)
        print('# Validation loss = {}'.format(val_loss))
        print('# Validation acc = {}'.format(val_acc))
        print(val_acc > best_acc)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(epoch_loss)
        val_loss_list.append(val_loss)
    return model, train_acc_list, val_acc_list, train_loss_list, val_loss_list


def train(model, dataset, epochs, path_to_weights, path_to_results):
    1. Initialize parameters of training session
    2. Build file parameters based on them
    3. Try to read file
    4. From file get start_epoch, weights
    5. Resume training (call train_num_epochs)
    6. Construct a dictionary:
    dictionary = {
        'model': modelname
        'dataset': dataset
        'labeltype': labeltype
        'spectype': spectype
        'specshape': specshape
        'preprocess': preprocess
        'padding': padding
        'tasks': tasks
        'subset': subset
        'accuracy': accuracy
        'loss': loss
    }

def validate(model, criterion, testloader, device):
    dataset_size = len(testloader.dataset)
    correct = 0
    total = 0
    model.eval()
    epoch_loss = 0.0
    for i, (data, target) in enumerate(testloader):
        t0 = time.time()
        data = data.to(device)
        target = target[0]
        target = target.to(device)
        with torch.no_grad():
            # run forward step
            predicted = model(data)
            loss = criterion(predicted, target)
            epoch_loss += loss.item() * data.size(0)
        _, pred_labels = torch.max(predicted.data, 1)
        total += target.size(0)
        correct += (pred_labels == target).sum().item()
    return epoch_loss / dataset_size, correct / total

if __name__ == '__main__':
    model = CNNFromPaper()
    x = torch.randn(1, 1, 64, 64)
    # Let's print it
    print(model(x).shape)

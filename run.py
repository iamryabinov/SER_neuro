import models
import torch
from torch import nn
from datasets import *
from consants import *
from torchsummary import summary

# Devices:
cuda = torch.device('cuda:0')
cpu = torch.device('cpu')
device = cuda

# Dataset and train-test loaders:
iemocap_no_prep_dataset = IemocapDataset(path=IEMOCAP_PATH,
                                         name='IEMOCAP with no preprocessing',
                                         spectrogram_shape=224)
iemocap_no_prep_trainloader, iemocap_no_prep_testloader = train_test_loaders(iemocap_no_prep_dataset,
                                                                             batch_size=64)

# Model:
model_iemocap = models.alexnet(num_classes=len(iemocap_no_prep_dataset.emotions_dict))
model_iemocap.to(device)

# Criterion, optimizer, number of epochs:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_iemocap.parameters())

if __name__ == '__main__':
    models.train_num_epochs(model=model_iemocap,
                            trainloader=iemocap_no_prep_trainloader, testloader=iemocap_no_prep_testloader,
                            device=device, criterion=criterion, optimizer=optimizer,
                            starting_epoch=0, ending_epoch=1)

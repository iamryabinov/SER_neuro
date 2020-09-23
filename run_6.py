import models
import torch
from torch import nn
from datasets import *
from consants import *
from torchsummary import summary
import pandas as pd

if __name__ == '__main__':
    # Devices:
    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = cpu

    # Dataset and train-test loaders:
    iemocap = IemocapDataset(
        pickle_folder=PATH_TO_JSONS, wavs_folder=IEMOCAP_PATH_TO_WAVS,
        base_name='IEMOCAP', label_type='four', preprocessing='true',
        spectrogram_shape=128, spectrogram_type='melspec')
    print('Loaded dataset successfully')
    iemocap_trainloader, iemocap_testloader = train_test_loaders(iemocap, batch_size=64)
    print('Train-test loaders ready')

    # Model:
    model_iemocap = models.alexnet(num_classes=len(iemocap.emotions_dict))
    model_iemocap.to(device)
    print('Model ready')

    # Criterion, optimizer, number of epochs:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_iemocap.parameters())
    print('Done!')
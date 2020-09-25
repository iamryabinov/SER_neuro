import models
import torch
from torch import nn
from datasets import *
from constants import *
from torchsummary import summary
import pandas as pd



if __name__ == '__main__':
    # Devices:
    cuda = torch.device('cuda:0')
    cpu = torch.device('cpu')
    device = cpu

    # Dataset and train-test loaders:
    iemocap = IemocapDataset(
        pickle_folder=PATH_TO_PICKLE, wavs_folder=IEMOCAP_PATH_TO_WAVS,
        base_name='IEMOCAP', label_type='four', preprocessing='true',
        spectrogram_shape=224, spectrogram_type='melspec')
    print('Loaded dataset successfully')
    iemocap_trainloader, iemocap_testloader = train_test_loaders(iemocap, batch_size=64)
    print('Train-test loaders ready')
    #
    # # Model:
    model_iemocap = models.alexnet(num_classes=len(iemocap.emotions_dict))
    model_iemocap.to(device)
    print('Model ready')
    #
    # # Criterion, optimizer, number of epochs:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_iemocap.parameters())
    print('Done!')
    model, train_acc_list, val_acc_list, train_loss_list, val_loss_list = models.train_num_epochs(model=model_iemocap,
                            trainloader=iemocap_trainloader, testloader=iemocap_testloader,
                            device=device, criterion=criterion, optimizer=optimizer,
                            starting_epoch=0, ending_epoch=50)
    dictionary = {
        'train acc': train_acc_list,
        'val acc': val_acc_list,
        'train loss': train_loss_list,
        'val loss': val_loss_list
    }
    df = pd.DataFrame(dictionary)
    df.to_csv('first_result_prep.csv', sep=';', index=False)
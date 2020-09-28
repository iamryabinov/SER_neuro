from models import *
import torch
from torch import nn
import datasets
from constants import *
from torchsummary import summary
import pandas as pd

if __name__ == '__main__':
    cpu = torch.device('cpu')
    device = cpu
    print('Devices ready')
    dataset_256 = datasets.iemocap_four_256_noprep_zeropad
    dataset_64 = datasets.iemocap_four_64_noprep_zeropad
    print('Datasets ready')
    model_alex = AlexNet(num_classes=len(dataset_256.emotions_dict))
    model_deepnet = PaperCnnDeepNet(num_classes=len(dataset_64.emotions_dict))
    print('Models ready')
    criterion = nn.CrossEntropyLoss()

    lr = 0.001
    optimizer_alex = torch.optim.Adam(model_alex.parameters(), lr=lr, amsgrad=True)
    scheduler_alex = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_alex, mode='min', factor=0.1, patience=5,
                                                                cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    optimizer_deepnet = torch.optim.Adam(model_deepnet.parameters(), lr=lr, amsgrad=True)
    scheduler_deepnet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_deepnet, mode='min', factor=0.1,
                                                                   patience=5,
                                                                   cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    print('Optimizers and schedulers ready')
    session_alex = TrainingSession(
        model=model_alex, dataset=dataset_256,
        criterion=criterion, optimizer=optimizer_alex, scheduler=scheduler_alex, num_epochs=100,
        batch_size=64, device=device,
        path_to_weights=WEIGHTS_FOLDER, path_to_results=RESULTS_FOLDER
    )
    session_deepnet = TrainingSession(
        model=model_deepnet, dataset=dataset_64,
        criterion=criterion, optimizer=optimizer_deepnet, scheduler=scheduler_deepnet, num_epochs=100,
        batch_size=256, device=device,
        path_to_weights=WEIGHTS_FOLDER, path_to_results=RESULTS_FOLDER
    )
    print('Training sessions ready')
    # session_deepnet.overfit_one_batch(100, 5)
    session_deepnet.execute()
    session_alex.execute()


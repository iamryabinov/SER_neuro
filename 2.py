import models.models_multi_task as md_multi
from models.multitask_training_session import *
import datasets.iemocap as ds
from constants import *
from torchsummary import summary
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd


train_ds = ds.IemocapDataset(  # Без препроцессинга, тренировочный
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='train', preprocessing=False,
    augmentation=True, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec', tasks=('emotion', 'speaker', 'gender')
)
test_ds = ds.IemocapDataset(  # Без препроцессинга, тестовый
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='test', preprocessing=False,
    augmentation=True, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec', tasks=('emotion', 'speaker', 'gender')
)

model = md_multi.AlexNetMultiTask(4, 10, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
ts = TrainingSession(name='AverageWeighting',
                      model=model,
                      train_dataset=train_ds,
                      test_dataset=test_ds,
                      criterion=criterion,
                      optimizer=optimizer,
                      num_epochs=300,
                      batch_size=32,
                      device=device,
                     path_to_weights=WEIGHTS_FOLDER,
                     path_to_results=RESULTS_FOLDER,
                    loss_weighter=averaged_sum  # Из файла multitask_training_session.py
                    )

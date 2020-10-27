from models.models_one_task_egemaps import AlexNetEgemaps
from  datasets.iemocap import IemocapDataset, train_test_loaders
from constants import *
from torchsummary import summary
import torch
import torch.nn as nn
import skorch
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from skorch.classifier import NeuralNetClassifier
import skorch.callbacks as callbacks
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


train_ds = IemocapDataset(  # Без препроцессинга, тренировочный
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='train', preprocessing=False,
    augmentation=True, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec', tasks='emotion', egemaps=True
)

X = [X for X, y in iter(train_ds)]
y = [y for X, y in iter(train_ds)]

_X = X[:100]
y_overfit = y[:100]
X_overfit = {
    'spectrogram': [],
    'egemaps': []
}

for sample in _X:
    X_overfit['spectrogram'].append(sample['spectrogram'])
    X_overfit['egemaps'].append(sample['egemaps'])

overfit_dataset = skorch.dataset.Dataset(X=X_overfit, y=y_overfit)
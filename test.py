from iemocap import IemocapDataset
from constants import *
import torch.nn as nn
import skorch
import skorch.dataset
from skorch.callbacks import *
import torch
import numpy as np
from torch.optim.lr_scheduler import CyclicLR

iemocap_four_64_noprep_zeropad = IemocapDataset(
    pickle_path=PATH_TO_PICKLE,
    wavs_path=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=64,
    preprocessing='false',
    padding='zero')
dataset = iemocap_four_64_noprep_zeropad
cpu = torch.device('cpu')
device = cpu
print('Models ready')
criterion = nn.CrossEntropyLoss
lr = 0.00001
optimizer = torch.optim.Adam

net = skorch.NeuralNetClassifier(module=model, criterion=criterion,
                                 train_split=skorch.dataset.CVSplit(5, stratified=True),
                                 optimizer=optimizer, lr=lr, max_epochs=1000, batch_size=64,
                                 dataset=dataset, device=cpu,
                                 )

x_train = np.array(X for X, y in iter(dataset))
y_train = np.array([y for X, y in iter(dataset)])

net.fit(X=x_train, y=y_train)

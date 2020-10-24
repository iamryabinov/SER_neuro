from iemocap_egemaps import EgemapsDataset
from constants import *
import torch.nn as nn
import skorch
import skorch.dataset
from skorch.callbacks import *
import torch
import numpy as np
from torch.optim.lr_scheduler import CyclicLR
from models_one_task import EgemapsPerceptron
import pandas as pd

filename = 'egemaps_perceptron_iemocap'
best_model_file_path = os.path.join(RESULTS_FOLDER, filename)
dataset = EgemapsDataset('IEMOCAP-4',
                         folder='E:\\Projects\\SER\\knn_svm\\datasets\\iemocap\\',
                         language='English',
                         label='four')
device = torch.device('cuda')
print('Models ready')
criterion = nn.CrossEntropyLoss
lr = 3e-4
optimizer = torch.optim.Adam
model = EgemapsPerceptron(num_classes=4)
callback_train_acc = EpochScoring(scoring="accuracy",
                                  lower_is_better=False,
                                  on_train=True,
                                  name='train_acc')
callback_save_best = Checkpoint(monitor='valid_acc_best',
                                f_params=None,
                                f_optimizer=None,
                                f_criterion=None,
                                f_history=None,
                                f_pickle=best_model_file_path,
                                event_name='event_cp')
callback_early_stop = EarlyStopping(monitor='valid_loss', patience=30,
                                    threshold_mode='rel', lower_is_better=True)
net = skorch.NeuralNetClassifier(module=model, criterion=criterion,
                                 train_split=skorch.dataset.CVSplit(10, stratified=True),
                                 optimizer=optimizer, lr=lr, max_epochs=150, batch_size=32,
                                 dataset=dataset, device=device, callbacks=[
        callback_train_acc,
        callback_early_stop,
        callback_save_best
    ]
                                 # optimizer__weight_decay = 3e-4,
                                 )


if __name__ == '__main__':
    x_train = np.array(X for X, y in iter(dataset))
    y_train = np.array([y for X, y in iter(dataset)])
    net.fit(X=x_train, y=y_train)



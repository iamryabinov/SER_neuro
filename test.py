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
net = skorch.NeuralNetClassifier(module=model, criterion=criterion,
                                 train_split=skorch.dataset.CVSplit(10, stratified=True),
                                 optimizer=optimizer, lr=lr, max_epochs=150, batch_size=32,
                                 dataset=dataset, device=device, callbacks=[
                                     callback_train_acc
                                    ]
                                 # optimizer__weight_decay = 3e-4,
                                 )
x_train = np.array(X for X, y in iter(dataset))
y_train = np.array([y for X, y in iter(dataset)])
net.fit(X=x_train, y=y_train)

def create_results_file(net, name):
    dfs_list = []
    for metric in ['accuracy', 'loss']:
        _dfs_list = []
        for subset in ['train', 'test']:
            results = net.history[:, f'{subset}_{metric}']
            df = pd.DataFrame(results, columns=['result'])
            df['epochs'] = np.arange(1, len(results) + 1)
            df['subset'] = subset
            _dfs_list.append(df)
        df = pd.concat(_dfs_list, ignore_index=True)
        df['metric'] = metric
        dfs_list.append(df)
    final_df = pd.concat(dfs_list, ignore_index=True)
    final_df.to_csv(os.path.join(RESULTS_FOLDER, f'{name}__result.csv'), sep=';', index=False)
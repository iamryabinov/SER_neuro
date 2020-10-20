import os
import pandas as pd
import torch
import torchvision.transforms as transforms


class EgemapsDataset(torch.utils.data.Dataset):
    emotions_dict = {
        'ang': 0,
        'hap': 1,
        'neu': 2,
        'sad': 3,
        'exc': 4,
        'fru': 5,
    }

    def __init__(self, name, folder, language, label):
        super(EgemapsDataset, self).__init__()
        self.name = name
        self.folder = folder
        self.language = language
        self.feature_folder = folder + 'features\\'
        self.path_to_feature_file = self.feature_folder + '_features_with_labels.csv'
        if not os.path.exists(self.path_to_feature_file):
            raise FileNotFoundError('Feature file not found!')
        self.features = FeatureFile(self.path_to_feature_file, label)
        self.label = label

    def __len__(self):
        return len(self.features.X)

    def __getitem__(self, idx):
        X = self.features.X[idx]
        X = torch.from_numpy(X).float()
        y = self.features.y[idx]
        y = self.emotions_dict[y]
        return X, y


class FeatureFile:
    def __init__(self, path, label):
        self.contents = pd.read_csv(path, delimiter=';')
        self.X, self.y = self._get_xy(label)

    def _get_xy(self, label):
        df = self.contents
        if label == 'four':
            df = df.loc[
                (df['label'] == 'neu') | (df['label'] == 'ang') | (df['label'] == 'hap') | (df['label'] == 'sad')]
        X = df.drop(['name', 'label', 'frameTime', 'label_2', 'label_3'], axis=1).values
        if label == 'original' or label == 'four':
            y = df[['label']].values.ravel()
        elif label == 'pos_neg_neu':
            y = df[['label_2']].values.ravel()
        elif label == 'negative_binary':
            y = df[['label_3']].values.ravel()
        else:
            raise ValueError('Unknown value for "label"')
        return X, y



iemocap_four_labels = EgemapsDataset('IEMOCAP-4',
                                     folder='E:\\Projects\\SER\\knn_svm\\datasets\\iemocap\\',
                                     language='English',
                                     label='four'
                                     )

if __name__ == '__main__':
    iemo = EgemapsDataset('IEMOCAP-4',
                                 folder='E:\\Projects\\SER\\knn_svm\\datasets\\iemocap\\',
                                 language='English',
                                 label='four')
    print(len(iemo))
    print(iemo[0])
    pass

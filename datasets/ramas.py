import librosa
import librosa.display
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import random_split
from constants import *
import cv2
import os
import matplotlib.pyplot as plt


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def get_paths_to_wavs(path_to_dataset_wavs):
    file_paths_list = []
    noise_file_path = ''
    for root, _dirs, files in os.walk(path_to_dataset_wavs):  # Iterate over files in directory
        if len(files) != 0:
            for f in files:
                if f.endswith('.wav'):
                    if 'noise' in f:
                        noise_file_path = os.path.join(root, f)
                    else:
                        file_paths_list.append(os.path.join(root, f))
                else:
                    continue
    if file_paths_list == []:
        raise FileNotFoundError('Returned empty list!')
    return file_paths_list, noise_file_path


class RamasDataset(torch.utils.data.Dataset):

    emotions_dict = {
        'Domination + Anger': 0,
        'Other': 1,
        'Angry': 0,
        'Disgusted': 1,
        'Happy': 2,
        'Neutral': 3,
        'Sad': 4,
        'Scared': 5,
        'Shame': 6,
        'Surprised': 7,
    }

    speakers_dict = {
        '10dec1': 0,
        '10dec2': 1,
        '14dec1': 2,
        '14dec2': 3,
        '15dec1': 4,
        '15dec2': 5,
        '16dec1': 6,
        '16dec2': 7,
        '19dec1': 8,
        '19dec2': 9,
        '22dec1': 10,
        '22dec2': 11
    }

    genders_dict = {
        '10dec1': 1,
        '10dec2': 0,
        '14dec1': 0,
        '14dec2': 1,
        '15dec1': 1,
        '15dec2': 0,
        '16dec1': 0,
        '16dec2': 1,
        '19dec1': 1,
        '19dec2': 0,
        '22dec1': 0,
        '22dec2': 1
    }

    def __init__(self, wavs_path, base_name,
                 spectrogram_shape=224,
                 augmentation=False, padding='zero', mode='train',  tasks='emotion', type='descrete'):
        super(RamasDataset, self).__init__()
        self.name = '{}_{}_{}'.format(
            base_name, spectrogram_shape, mode)
        print('============= INITIALIZING DATASET {} ==============='.format(self.name))
        self.mode = mode
        self.spectrogram_shape = spectrogram_shape
        self.tasks = tasks
        self.padding = padding
        self.augmentation = augmentation
        self.folder = os.path.join(wavs_path, mode)
        if not os.path.exists(self.folder):
            raise OSError('Path not found!')
        self.type = type
        self.paths_to_wavs, _ = self.get_paths_to_wavs(self.folder)
#         if self.type == 'descrete':
#             class_weights = [0.8295337851714931, 0.8979256520846169, 0.7732593961799137, 0.9301704662148285,
#                              0.8782090778393921, 0.8016019716574245, 0.9767919490655166, 0.9125077017868145]
#             self.class_weights = torch.FloatTensor(class_weights).cuda()
#         elif self.type == 'binary':
#             class_weights = [0.8618286182861828, 0.13817138171381715]
#             self.class_weights = torch.FloatTensor(class_weights).cuda()
#         else:
#             raise ValueError('Unknown value for type! Should be either "descrete" or "binary"!')
        print('============================ SUCCESS! =========================')

    def get_paths_to_wavs(self, path_to_dataset_wavs):
        file_paths_list = []
        noise_file_path = ''
        for root, _dirs, files in os.walk(path_to_dataset_wavs):  # Iterate over files in directory
            for f in files:
                if f.endswith('.wav'):
                    file_paths_list.append(os.path.join(root, f))
        if not file_paths_list:
            raise FileNotFoundError('Returned empty list!')
        return file_paths_list, noise_file_path

    def read_audio(self, path_to_wav):
        """
        Read .wav file using librosa, keeping orignal framerate
        """
        y, sr = librosa.load(path_to_wav, sr=None)
        return (y, sr)

    def get_labels(self, path):
        file_name = os.path.split(path)[1]
        file_name = file_name[:-4]
        date, _, speaker_id, _, emotion_descrete, emotion_binary = file_name.split('_')
        speaker = date + speaker_id
        gender = date + speaker_id
        descrete_label = self.emotions_dict[emotion_descrete]
        binary_label = 'Angry' if emotion_descrete == 'Angry' else 'Other'
        binary_label = self.emotions_dict[binary_label]
        speaker = self.speakers_dict[speaker]
        gender = self.speakers_dict[gender]
        return descrete_label, binary_label, speaker, gender

    def __len__(self):
        return len(self.paths_to_wavs)

    def __getitem__(self, idx):
        path_to_item = self.paths_to_wavs[idx]
        wav = self.read_audio(path_to_item)
        spec = self.make_spectrogram(wav)
        if self.mode == 'train':
            if self.augmentation:
                spec = self.augment(spec)
            else:
                spec = self.unify_size(spec)
        elif self.mode == 'test':
            spec = self.unify_size(spec)
        else:
            raise ValueError('Unknown value for mode: should be either "train" or "test"!')
#         print(f'{idx + 1}/{self.__len__()} || {path_to_item}: {spec.shape}')
        img = scale_minmax(spec, 0, 255).astype(np.uint8)  # min-max scale to fit inside 8-bit range
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img_shape = self.spectrogram_shape
        img = cv2.resize(img, dsize=(img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
        normalize_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.225])
        ])
        img = normalize_image(img)
        data = img
        descrete_label, binary_label, speaker, gender = self.get_labels(path_to_item)
        # labels = emotion, speaker, gender
        emotion = descrete_label if self.type == 'descrete' else binary_label
        if self.tasks == 'emotion':
            labels = emotion
        elif self.tasks == 'multi':
            labels = emotion, speaker, gender
        return data, labels

    def make_spectrogram(self, wav):
        """
        Create an ordinary or mel-scaled x1, given vaw (y, sr).
        self.spectrogram_type states if ordinary or mel x1 will be created.
        All spectrograms are log(dB)-scaled and min-max normalized.
        In order to keep the shape constant, random cropping or zero-padding is performed.
        """
        y, sr = wav
        n_mels = self.spectrogram_shape
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        spec = librosa.power_to_db(spec)
        return spec

    def augment(self, spec):
        """
        Random augmentation of short spectrograms: random zero-padding or random cropping.
        Makes the x1 square-shaped
        :param spec:
        :return:
        """
        rows, cols = spec.shape
        desired_shape = rows
        diff = cols - desired_shape
        if diff > 0:  # Random crop
            return self.random_crop(spec)
        elif diff < 0:  # Random augmentation
            # Return random integers from low (inclusive) to high(exclusive).
            dice = np.random.randint(1, 5)
            if dice == 1:
                return self.zero_pad(spec)
            else:
                repeat = np.random.randint(1, dice)
                spec = self.repeat(spec, repeat)
            return self.augment(spec)
        else:
            return spec

    def unify_size(self, spec):
        rows, cols = spec.shape
        desired_shape = rows
        diff = cols - desired_shape
        while not diff == 0:
            if diff > 0:  # Random crop
                spec = self.random_crop(spec)
                diff = spec.shape[1] - desired_shape
            elif diff < 0:  # Pad
                if self.padding == 'zero':  # Random zero-pad
                    spec = self.zero_pad(spec)
                    diff = spec.shape[1] - desired_shape
                elif self.padding == 'repeat':  # Pad x1 with itself
                    spec = self.repeat(spec, 2)
                    diff = spec.shape[1] - desired_shape
                else:
                    raise ValueError('Unknown value for padding: should be either "zero" or "repeat"!')
        return spec

    def random_crop(self, spec):
        rows, cols = spec.shape
        desired_shape = rows
        diff = cols - desired_shape
        seed = RANDOM_SEED if self.mode == 'test' else None  # To get the same spectrograms on the test set!!!
        np.random.seed(seed)
        beginning_col = np.random.randint(0, diff + 1)
        spec = spec[:, beginning_col:beginning_col + desired_shape]
        spec = scale_minmax(spec, 0, 255).astype(np.uint8)
        return spec

    def zero_pad(self, spec):
        rows, cols = spec.shape
        desired_shape = rows
        spec = scale_minmax(spec, 0, 255).astype(np.uint8)
        zeros = np.zeros((rows, desired_shape), dtype=np.uint8)
        seed = RANDOM_SEED if self.mode == 'test' else None  # To get the same spectrograms on the test set!!!
        np.random.seed(seed)
        beginning_col = np.random.randint(0, desired_shape - cols + 1)
        zeros[..., beginning_col:beginning_col + cols] = spec
        spec = zeros
        return spec

    def repeat(self, spec, times):
        return np.tile(spec, times)

    def show_image(self, idx, **kwargs):
        """
        Function that shows an image in form of mpl axes
        """
        img, labels = self.__getitem__(idx)
        img = img.numpy()
        img = np.squeeze(img, axis=0)
        ax = plt.imshow(img, **kwargs)
        return ax



if __name__ == '__main__':
    ramas_train = RamasDataset(wavs_path=RAMAS_PATH_TO_WAVS,
                               base_name='RAMAS-DomSub',
                               spectrogram_shape=224,
                               augmentation=False,
                               padding='zero',
                               mode='train',
                               tasks='emotion')
    ax = ramas_train.show_image(4)
    plt.show()
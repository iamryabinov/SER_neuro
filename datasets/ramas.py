import librosa
import librosa.display
import torch
import torchvision.transforms as transforms
import numpy as np
import noisereduce as nr
from torch.utils.data import random_split
from constants import *
import pickle
import cv2
import os
import glob
import pandas as pd
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
        'Domination': 0,
        'Submission': 1,
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

    def __init__(self, pickle_path, wavs_path, egemaps_path, path_for_parser,
                 base_name, label_type='original', spectrogram_type='melspec', spectrogram_shape=128,
                 preprocessing=False, augmentation=False, padding='zero', mode='train', tasks=['emotion']):
        super(RamasDataset, self).__init__()
        self.name = '{}_{}_prep-{}_{}_{}'.format(
            base_name, label_type, str(preprocessing).lower(), spectrogram_shape, mode)
        self.label_type = label_type
        print('============= INITIALIZING DATASET {} ==============='.format(self.name))
        self.preprocessing = preprocessing
        self.mode = mode
        self.spectrogram_shape = spectrogram_shape
        self.spectrogram_type = spectrogram_type
        self.tasks = tasks
        self.padding = padding
        self.augmentation = augmentation
        path = os.path.join(wavs_path, mode)
        pkl_path = '{}{}.pkl'.format(pickle_path, self.name)
        try:
            dictionary = pickle.load(open(pkl_path, "rb"))
            self.noise = np.array(dictionary['noise'], dtype=np.float32)
            self.sr = dictionary['sr']
        except FileNotFoundError:
            self.parsed_dict = self.get_parsed_dict(path_for_parser)
            paths_to_wavs_list, path_to_noise = self.my_get_paths_to_wavs(path)
            # Todo: we could actually even implement egemaps extraction here
            try:
                df = pd.read_csv(egemaps_path, delimiter=';')
                names_list = df['name'].values.tolist()
                names_list = [name[1:-1] for name in names_list]
                df['name'] = names_list
                self.egemaps_df = df
            except FileNotFoundError:
                self.egemaps_df = None
            self.noise, self.sr = self.read_audio(path_to_noise)
            dictionary = self.create_pickle_file(pkl_path, paths_to_wavs_list)
        self.files = dictionary['files']
        del dictionary
        print('=========================== SUCCESS! ====================================')

    def get_parsed_dict(self, path):
        print('Parsing dataset files...')
        return IEMOCAP_Dataset_parser(path).read_dataset()

    def my_get_paths_to_wavs(self, path):
        """
        Depending on labeling type (original or four),
        create and return list of paths to *.wav files.
        :param path: pickle_path to folder with all needed *.wav files
                     and file noise.wav, containing noise sample
        :return: list with paths to *.wav files and to noise.wav
        """
        if self.label_type == 'original':
            # Just get all files
            return get_paths_to_wavs(path)
        elif self.label_type == 'four':
            # Filter out emotions Excitement and Frustration, leaving only
            # anger, happiness, neutral, sadness.
            new_paths_to_wavs = []
            paths_to_wavs, path_to_noise = get_paths_to_wavs(path)
            for file in paths_to_wavs:
                file_name = os.path.split(file)[1]
                emotion_label = self.get_emotion_label(file_name)
                if emotion_label in ('ang', 'hap', 'neu', 'sad'):
                    new_paths_to_wavs.append(file)
            return new_paths_to_wavs, path_to_noise
        else:
            raise ValueError('Unknown label type! Should be either "original" for all samples, or "four" for anger, '
                             'happiness, neutral, sadness')

    def read_audio(self, path_to_wav):
        """
        Read .wav file using librosa, keeping orignal framerate
        """
        y, sr = librosa.load(path_to_wav, sr=None)
        return (y, sr)

    def create_pickle_file(self, path_to_pkl, paths_to_wavs_list):
        noise, sr = self.noise, self.sr
        dictionary = {
            'name': self.name,
            'noise': noise.tolist(),
            'sr': sr,
            'files': self.create_files_dicts_list(paths_to_wavs_list)
        }
        print('Writing file... ')
        pickle.dump(dictionary, open(path_to_pkl, "wb"))
        return dictionary

    def create_files_dicts_list(self, paths_to_wavs_list):
        files_dicts_list = []
        step = 1
        for file_path in paths_to_wavs_list:
            print('============================================')
            print('File {} of {}: {}'.format(step, len(paths_to_wavs_list), file_path))
            files_dicts_list.append(self.create_one_file_dict(file_path))
            print('Done!')
            step += 1
        return files_dicts_list

    def create_one_file_dict(self, file_path):
        print('Reading Audio...')
        y, sr = self.read_audio(file_path)
        if self.preprocessing:
            print('Preprocessing...')
            y, sr = self.preprocess(y, sr)
        file_name = os.path.split(file_path)[1]
        print('Making spectrogram...')
        spec = self.make_spectrogram((y, sr))
        print('Extracting egemaps...')
        egemaps = self.get_egemaps(file_name)
        print('Getting labels...')
        speaker, gender, valence, action, dominance = self.get_labels(file_name)
        emotion = self.get_emotion_label(file_name)
        files_dict = {
            'name': file_name,
            'spectrogram': spec,
            'egemaps': egemaps,
            'speaker': speaker,
            'gender': gender,
            'emotion': emotion,
            'valence': valence,
            'action': action,
            'dominance': dominance
        }
        print('Appending...')
        return files_dict

    def preprocess(self, y, sr):
        wav = y, sr
        noise_sample = self.noise
        preprocessed = transforms.Compose([
            Denoiser(noise_sample, 2),
            RemoveSilence(40)
        ])
        y, sr = preprocessed(wav)
        return y, sr

    def get_egemaps(self, file_name):
        """
        eGeMAPS feature set for this file
        """
        file_name = file_name[:-4]
        df = self.egemaps_df
        if df is None:
            return ValueError('No egemaps file was found!')
        else:
            egemaps = df.loc[df['name'] == file_name].drop('name', axis=1).values
            return egemaps

    def get_emotion_label(self, file_name):
        """
        Parse the filename, return emotion label
        """
        file_name = file_name[:-4]
        emotion_name = file_name.split('_')[-1]  # the last is a position of emotion code
        return emotion_name

    def get_labels(self, file_name):
        file_name = file_name[:-8]
        parsed_dict = self.parsed_dict[file_name]
        speaker = parsed_dict["Speaker-Id"]
        gender = parsed_dict["Gender"]
        valence = float(parsed_dict["Valence"])
        action = float(parsed_dict["Action"])
        dominance = float(parsed_dict["Dominance"])
        valence = 'negative' if valence <= 2 else 'high' if valence > 3.5 else 'medium'
        action = 'negative' if action <= 2 else 'high' if action > 3.5 else 'medium'
        dominance = 'negative' if dominance <= 2 else 'high' if dominance > 3.5 else 'medium'
        return speaker, gender, valence, action, dominance

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_instance = self.files[idx]
        spec = file_instance['spectrogram']
        if self.mode == 'train':
            if self.augmentation:
                spec = self.augment(spec)
            else:
                spec = self.unify_size(spec)
        elif self.mode == 'test':
            spec = self.unify_size(spec)
        else:
            raise ValueError('Unknown value for mode: should be either "train" or "test"!')
        img = scale_minmax(spec, 0, 255).astype(np.uint8)  # min-max scale to fit inside 8-bit range
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img_shape = self.spectrogram_shape
        img = cv2.resize(img, dsize=(img_shape, img_shape), interpolation=cv2.INTER_CUBIC)
        normalize_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.225])
        ])
        img = normalize_image(img)
        labels = []
        for task in self.tasks:
            labels.append(file_instance[task])
        data = img
        target = labels[0]
        return data, self.emotions_dict[target]

    def make_spectrogram(self, wav):
        """
        Create an ordinary or mel-scaled spectrogram, given vaw (y, sr).
        self.spectrogram_type states if ordinary or mel spectrogram will be created.
        All spectrograms are log(dB)-scaled and min-max normalized.
        In order to keep the shape constant, random cropping or zero-padding is performed.
        """
        y, sr = wav
        shape = self.spectrogram_shape
        if shape < 100:
            n_mels = 128
            n_fft = 512
            hop_length = 512
        elif shape < 200:
            n_mels = 256
            n_fft = 1024
            hop_length = 256
        else:
            n_mels = 512
            n_fft = 1408
            hop_length = 256
        if self.spectrogram_type == 'melspec':
            spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                                  hop_length=hop_length,
                                                  n_fft=n_fft, n_mels=n_mels)
            spec = librosa.power_to_db(spec)
        elif self.spectrogram_type == 'spec':
            spec = np.abs(librosa.core.stft(y=y, n_fft=shape * 4, hop_length=hop_length))
            spec = librosa.amplitude_to_db(spec, ref=np.max)
        else:
            raise ValueError('Unknown value for spectrogram_type: should be either melspec or spec!')
        return spec

    def augment(self, spec):
        """
        Random augmentation of short spectrograms: random zero-padding or random cropping.
        Makes the spectrogram square-shaped
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
                elif self.padding == 'repeat':  # Pad spectrogram with itself
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
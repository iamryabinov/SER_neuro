import os
import librosa
import librosa.display
import torch
import torchvision.transforms as transforms
import numpy as np
import noisereduce as nr
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from consants import *
import json


def get_paths_to_wavs(path_to_dataset):
    file_paths_list = []
    noise_file_path = ''
    for root, _dirs, files in os.walk(path_to_dataset):  # Iterate over files in directory
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


class Denoiser:
    """
    Denoise using spectral gating. Requires noise sample.

    Threshold parameter affects how many standard deviations louder than the mean dB of the noise
    (at each frequency level) to be considered signal
    """

    def __init__(self, noise_sample, threshold):
        self.noise = noise_sample[0]
        self.threshold = threshold

    def __call__(self, wav):
        y, sr = wav
        noise_reduced = nr.reduce_noise(audio_clip=y, noise_clip=self.noise, prop_decrease=1, verbose=False,
                                        n_std_thresh=self.threshold, use_tensorflow=False)
        return noise_reduced, sr


class RemoveSilence:
    """
    Remove silent parts (threshold parameter is required to know which parts to consider silent)
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, wav):
        y, sr = wav
        nonsilent_y = []
        silent = librosa.effects.split(y, top_db=self.threshold, frame_length=2048, hop_length=512)
        for beginning, end in silent:
            nonsilent_part = y[beginning: end]
            nonsilent_y = np.concatenate((nonsilent_y, nonsilent_part))
        return nonsilent_y, sr



class IemocapDataset(torch.utils.data.Dataset):
    emotions_dict = {
        'exc': 0,
        'sad': 1,
        'fru': 2,
        'hap': 3,
        'neu': 4,
        'ang': 5,
    }

    def __init__(self, path, base_name, label_type='original',
                 spectrogram_type='melspec', spectrogram_shape=128, 
                 preprocessing='false', tasks=['emotion']):
        super(IemocapDataset, self).__init__()
        self.name = '{}_{}_prep-{}_{}_{}'.format(
            base_name, label_type, preprocessing, spectrogram_type, spectrogram_shape)
        path_to_json = '{}\\{}.json'.format(path, self.name)
        try:
            dictionary = self.read_json(path_to_json)
        except FileNotFoundError:
            dictionary = self.create_json_file(path_to_json)
        self.files = dictionary['files']
        self.label_type = label_type
        self.paths_to_wavs_list, self.path_to_noise = self.my_get_paths_to_wavs(path)
        self.preprocessing = preprocessing
        self.spectrogram_shape = spectrogram_shape
        self.spectrogram_type = spectrogram_type
        self.tasks = tasks
       

    def read_json(self, path_to_json):
        with open(path_to_json, 'r') as json_file:
            dictionary = json.load(json_file)
        return dictionary

    def create_json_file(self, path_to_json):
        noise, sr = self.read_audio(self.path_to_noise)
        dictionary = {
            'name': self.name, 
            'noise': noise,
            'sr': sr,
            'files': self.create_files_dicts_list()
            }
        with open(path_to_json, 'w') as json_file:
            json.dump(dictionary, json_file)
        return dictionary

    def create_files_dicts_list(self):
        files_dicts_list = []
        for file_path in self.paths_to_wavs_list:
            files_dicts_list.append(self.create_one_file_dict(file_path))
        return files_dicts_list

    def create_one_file_dict(self, file_path):
        y, sr = self.read_audio(file_path)
        if self.preprocessing == 'true':
            raise NotImplementedError('Preprocessing is not currently implemented!')
            # y, sr = self.preprocess(y, sr)
        shape = self.spectrogram_shape
        spectrogram = self.make_spectrogram((y, sr), shape)
        file_name = os.path.split(file_path)[1]
        egemaps = self.get_egemaps(file_path)
        emotion_label = self.get_emotion_label(file_path)
        gender_label = self.get_gender_label(file_path)
        speaker_label = self.get_speaker_label(file_path)
        files_dict = { 
            'name': file_name,
            'path': path,
            'y': y.tolist(),
            'spectrogram': spectrogram.tolist(),
            'egemaps': egemaps,
            'emotion': emotion_label,
            'gender': gender_label,
            'speaker': speaker_label
            }
        return files_dict

    def get_egemaps(self, file_path):
        """
        eGeMAPS feature set for this file
        Currently not implemented, todo!
        """
        return None

    def preprocess(self, y, sr):
        wav = y, sr
        noise_sample = self.read_audio(self.path_to_noise[0])
        preprocess = transforms.Compose([
                                        Denoiser(noise_sample, 25), 
                                        RemoveSilence(25)
                                        ])
        y, sr = preprocess(wav)
        return y, sr

    def my_get_paths_to_wavs(self, path):
        """
        Depending on labeling type (original or four), 
        create and return list of paths to *.wav files.
        :param path: path to folder with all needed *.wav files 
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
                emotion_label = self.get_emotion_label(file)
                if emotion_label in (1, 3, 4, 5):
                    new_paths_to_wavs.append(file)
        return new_paths_to_wavs, path_to_noise

    def get_emotion_label(self, path_to_file):
        """
        Parse the filename, return emotion label
        """
        file_name = os.path.split(path_to_file)[1]
        file_name = file_name[:-4]
        emotion_name = file_name.split('_')[-1]  # the last is a position of emotion code
        return self.emotions_dict[emotion_name]
    
    def get_gender_label(self, path_to_file):
        """
        Todo!
        """
        return None

    def get_speaker_label(self, path_to_file):
        """
        Todo!
        """
        return None

    def __len__(self):
        return len(self.files)

    def read_audio(self, path_to_wav):
        """
        Read .wav file using librosa, keeping orignal framerate
        """
        y, sr = librosa.load(path_to_wav, sr=None)
        return (y, sr)

    def make_spectrogram(self, wav, shape, hop_length=256):
        """
        Create an ordinary or mel-scaled spectrogram, given vaw (y, sr).
        self.spectrogram_type states if ordinary or mel spectrogram will be created.
        All spectrograms are log(dB)-scaled and min-max normalized.
        In order to keep the shape constant, random cropping or zero-padding is performed.
        """
        y, sr = wav
        if self.spectrogram_type == 'melspec':
            spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, 
                                                  n_fft=1280, n_mels=shape)
            spec = librosa.power_to_db(spec)
            spec = librosa.util.normalize(spec) + 1
        elif self.spectrogram_type == 'spec':
            raise NotImplementedError('This spectrogram type is not currently implemented!')
        else:
            raise ValueError('Unknown value for spectrogram_type: should be either melspec or spec!')
        rows, cols = spec.shape
        diff = cols - shape
        if diff > 0:  # Random crop
            beginning_col = np.random.randint(diff)
            spec = spec[:, beginning_col:beginning_col + shape]
        elif diff < 0:  # Random zero-pad
            zeros = np.zeros((shape, shape), dtype=np.float32)
            beginning_col = np.random.randint(shape - cols)
            zeros[..., beginning_col:beginning_col + cols] = spec
            spec = zeros
        return spec


    def __getitem__(self, idx):
        file_instance = self.files[idx]
        spec = np.array(file_instance['spectrogram'], dtype=np.float32)
        labels = []
        for task in self.tasks:
            labels.append(file_instance[task])
        return spec, labels


class RavdessDataset(IemocapDataset):
    emotions_dict = {
        0: 'neutral',
        1: 'calm',
        2: 'happiness',
        3: 'sadness',
        4: 'anger',
        5: 'fear',
        6: 'disgust',
        7: 'surprise'
    }

    def get_emotion_label(self, path_to_file):
        """
        Parse the filename, which has the following pattern:
        modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        e.g., '02-01-06-01-02-01-12.wav'
        """
        file_name = os.path.split(path_to_file)[1]
        file_name = file_name[:-4]
        class_label = int(file_name.split('-')[2]) - 1  # 2 is a number of emotion code
        return class_label

    def get_gender_label(self, path_to_file):
        return None

    def get_speaker_label(self, path_to_file):
        return None



def train_test_loaders(dataset, validation_ratio=0.2, **kwargs):
    """
    Create train and test DataLoaders
    :param kwargs: keyword arguments for DataLoader
    :return: train and test loaders
    """
    dataset_size = len(dataset)
    test_size = int(np.floor(validation_ratio * dataset_size))
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, (train_size, test_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    return train_loader, test_loader



if __name__ == '__main__':
    iemocap = IemocapDataset(
        path=IEMOCAP_PATH, base_name='IEMOCAP', label_type='four', 
        spectrogram_shape=224, spectrogram_type='melspec')
    for i in range(len(iemocap)):
        spec, [label] = iemocap[i]
        print(i, ' ', type(spec), ' ', spec.shape, ' ', label)
        if i % 5 ==0:
            librosa.display.specshow(spec, cmap='magma')
            plt.show()

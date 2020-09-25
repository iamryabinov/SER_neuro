import os
import librosa
import librosa.display
import torch
import torchvision.transforms as transforms
import numpy as np
import noisereduce as nr
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from constants import *
import json
from sys import getsizeof
import pickle
from PIL import Image
import cv2



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


class Denoiser:
    """
    Denoise using spectral gating. Requires noise sample.

    Threshold parameter affects how many standard deviations louder than the mean dB of the noise
    (at each frequency level) to be considered signal
    """

    def __init__(self, noise_sample, threshold):
        self.noise = noise_sample
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

    def __init__(self, pickle_folder, wavs_folder, base_name, label_type='original',
                 spectrogram_type='melspec', spectrogram_shape=128,
                 preprocessing='false', padding='zero', tasks=['emotion']):
        super(IemocapDataset, self).__init__()
        self.name = '{}_{}_prep-{}_{}_{}_{}'.format(
            base_name, label_type, preprocessing, spectrogram_type, spectrogram_shape, padding)
        self.label_type = label_type
        print('============= INITIALIZING DATASET {} ==============='.format(self.name))
        self.preprocessing = preprocessing
        self.spectrogram_shape = spectrogram_shape
        self.spectrogram_type = spectrogram_type
        self.tasks = tasks
        self.padding = padding
        pkl_path = '{}\\{}.pkl'.format(pickle_folder, self.name)
        try:
            dictionary = pickle.load(open(pkl_path, "rb"))
            self.noise = np.array(dictionary['noise'], dtype=np.float32)
            self.sr = dictionary['sr']
        except FileNotFoundError:
            paths_to_wavs_list, path_to_noise = self.my_get_paths_to_wavs(wavs_folder)
            self.noise, self.sr = self.read_audio(path_to_noise)
            dictionary = self.create_pickle_file(pkl_path, paths_to_wavs_list)
        self.files = dictionary['files']
        del dictionary
        print('=========================== SUCCESS! ====================================')

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
        print(getsizeof(files_dicts_list))
        return files_dicts_list

    def create_one_file_dict(self, file_path):
        print('Reading Audio...')
        y, sr = self.read_audio(file_path)
        if self.preprocessing == 'true':
            print('Preprocessing...')
            y, sr = self.preprocess(y, sr)
        file_name = os.path.split(file_path)[1]
        print('Making spectrogram...')
        spec = self.make_spectrogram((y, sr))
        print('Extracting egemaps...')
        egemaps = self.get_egemaps(file_path)
        print('Getting emotion, gender and speaker labels...')
        emotion_label = self.get_emotion_label(file_path)
        gender_label = self.get_gender_label(file_path)
        speaker_label = self.get_speaker_label(file_path)
        files_dict = {
            'name': file_name,
            'spectrogram': spec,
            'egemaps': egemaps,
            'emotion': emotion_label,
            'gender': gender_label,
            'speaker': speaker_label
            }
        print('Appending...')
        return files_dict

    def get_egemaps(self, file_path):
        """
        eGeMAPS feature set for this file
        Currently not implemented, todo!
        """
        return None

    def preprocess(self, y, sr):
        wav = y, sr
        noise_sample = self.noise
        preprocess = transforms.Compose([
                                        Denoiser(noise_sample, 1.7),
                                        RemoveSilence(30)
                                        ])
        y, sr = preprocess(wav)
        return y, sr

    def my_get_paths_to_wavs(self, path):
        """
        Depending on labeling type (original or four), 
        create and return list of paths to *.wav files.
        :param path: pickle_folder to folder with all needed *.wav files
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
        else:
            raise ValueError('Unknown label type! Should be either "original" for all samples, or "four" for anger, '
                             'happiness, neutral, sadness')

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

    def make_spectrogram(self, wav, hop_length=256):
        """
        Create an ordinary or mel-scaled spectrogram, given vaw (y, sr).
        self.spectrogram_type states if ordinary or mel spectrogram will be created.
        All spectrograms are log(dB)-scaled and min-max normalized.
        In order to keep the shape constant, random cropping or zero-padding is performed.
        """
        y, sr = wav
        shape = self.spectrogram_shape
        if shape < 100:
            a = 128
            aspect = 4
        elif shape < 200:
            a = 256
            aspect = 4
        else:
            a = 512
            aspect = 2.75
        if self.spectrogram_type == 'melspec':
            spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=192,
                                                  n_fft=int(a*aspect), n_mels=a)
            spec = librosa.power_to_db(spec)
        elif self.spectrogram_type == 'spec':
            spec = np.abs(librosa.core.stft(y=y, n_fft=shape*4, hop_length=192))
            spec = librosa.amplitude_to_db(spec, ref=np.max)
        else:
            raise ValueError('Unknown value for spectrogram_type: should be either melspec or spec!')
        rows, cols = spec.shape
        diff = cols - shape
        while not diff == 0:
            if diff > 0:  # Random crop
                beginning_col = np.random.randint(diff)
                spec = spec[:, beginning_col:beginning_col + shape]
            elif diff < 0:  # Pad
                if self.padding == 'zero': # Random zero-pad
                    spec = scale_minmax(spec, 0, 255).astype(np.uint8)
                    zeros = np.zeros((rows, shape), dtype=np.uint8)
                    beginning_col = np.random.randint(shape - cols)
                    zeros[..., beginning_col:beginning_col + cols] = spec
                    spec = zeros
                elif self.padding == 'repeat':  # Pad spectrogram with itself
                    spec = np.concatenate([spec, spec], axis=1)
                else: 
                    raise ValueError('Unknown value for padding: should be either "zero" or "repeat"!')
            diff = spec.shape[1] - shape
        # min-max scale to fit inside 8-bit range
        img = scale_minmax(spec, 0, 255).astype(np.uint8)
        # img = spec
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        # img = 255 - img  # invert. make black==more energy
        img = cv2.resize(img, dsize=(shape, shape), interpolation=cv2.INTER_CUBIC)
        return img


    def __getitem__(self, idx):
        file_instance = self.files[idx]
        spec = file_instance['spectrogram']
        return spec
        # spec = np.expand_dims(spec, axis=0)
        # labels = []
        # for task in self.tasks:
        #     labels.append(file_instance[task])
        # return torch.from_numpy(spec).float(), labels


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
    train_dataset, test_dataset = random_split(dataset, (train_size, test_size),
                                               generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    return train_loader, test_loader



iemocap_original_64_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape= 64,
    preprocessing='false',
    padding='zero')
iemocap_original_64_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=64,
    preprocessing='true',
    padding='zero')
iemocap_original_64_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=64,
    preprocessing='false',
    padding='repeat')
iemocap_original_64_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=64,
    preprocessing='true',
    padding='repeat')
iemocap_four_64_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=64,
    preprocessing='false',
    padding='repeat')
iemocap_four_64_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=64,
    preprocessing='true',
    padding='zeropad')
iemocap_four_64_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=64,
    preprocessing='false',
    padding='repeat')
iemocap_four_64_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=64,
    preprocessing='true',
    padding='repeat')

iemocap_original_128_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=128,
    preprocessing='false',
    padding='zero')
iemocap_original_128_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=128,
    preprocessing='true',
    padding='zero')
iemocap_original_128_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=128,
    preprocessing='false',
    padding='repeat')
iemocap_original_128_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=128,
    preprocessing='true',
    padding='repeat')
iemocap_four_128_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape= 128,
    preprocessing='false',
    padding='zero')
iemocap_four_128_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=128,
    preprocessing='true',
    padding='zero')
iemocap_four_128_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=128,
    preprocessing='false',
    padding='repeat')
iemocap_four_128_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=128,
    preprocessing='true',
    padding='repeat')

iemocap_original_256_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=256,
    preprocessing='false',
    padding='zero')
iemocap_original_256_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=256,
    preprocessing='true',
    padding='zero')
iemocap_original_256_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=256,
    preprocessing='false',
    padding='repeat')
iemocap_original_256_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='original',
    spectrogram_shape=256,
    preprocessing='true',
    padding='repeat')
iemocap_four_256_noprep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape= 256,
    preprocessing='false',
    padding='zero')
iemocap_four_256_prep_zeropad = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=256,
    preprocessing='true',
    padding='zero')
iemocap_four_256_noprep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=256,
    preprocessing='false',
    padding='repeat')
iemocap_four_256_prep_repeat = IemocapDataset(
    pickle_folder=PATH_TO_PICKLE,
    wavs_folder=IEMOCAP_PATH_TO_WAVS,
    base_name='IEMOCAP',
    label_type='four',
    spectrogram_shape=256,
    preprocessing='true',
    padding='repeat')

if __name__ == '__main__':
    pass
import os
import librosa
import librosa.display
import torch
import numpy as np
import noisereduce as nr
from torch.utils.data.sampler import SubsetRandomSampler


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


class RandomPitchShift:
    def __call__(self, wav):
        raise NotImplementedError


class IemocapDataset(torch.utils.data.Dataset):
    emotions_dict = {
        'exc': 0,
        'sad': 1,
        'fru': 2,
        'hap': 3,
        'neu': 4,
        'ang': 5,
    }

    def __init__(self, path, name, label_type='original',
                 spectrogram_type='mel', spectrogram_shape=128, transform=None):
        super(IemocapDataset, self).__init__()
        self.name = name
        self.path = path
        self.label_type = label_type
        self.paths_to_wavs_list, _ = self.my_get_paths_to_wavs(self.path)
        self.transform = transform
        self.spectrogram_shape = spectrogram_shape
        self.spectrogram_type = spectrogram_type

    def my_get_paths_to_wavs(self, path):
        if self.label_type == 'original':
            return get_paths_to_wavs(path)
        elif self.label_type == 'four':
            new_paths_to_wavs = []
            paths_to_wavs, path_to_noise = get_paths_to_wavs(path)
            for file in paths_to_wavs:
                emotion_label = self.get_emotion_label(file)
                if emotion_label in (1, 3, 4, 5):
                    new_paths_to_wavs.append(file)
        return new_paths_to_wavs, path_to_noise

    def get_emotion_label(self, path_to_file):
        file_name = os.path.split(path_to_file)[1]
        file_name = file_name[:-4]
        emotion_name = file_name.split('_')[-1]  # the last is a position of emotion code
        return self.emotions_dict[emotion_name]

    def __len__(self):
        return len(self.paths_to_wavs_list)

    def read_audio(self, path_to_wav):
        """
        Read .wav file using librosa, NOT keeping orignal framerate
        """
        y, sr = librosa.load(path_to_wav)
        return (y, sr)

    def make_spectrogram(self, wav, shape):
        """
        Ordinary spectrogram (dB)
        """
        raise NotImplementedError

    def make_melspectrogram(self, wav, shape, hop_length=256):
        """
        Mel-scaled spectrogram (dB)

        In order to keep the shape constant, random cropping or zero-padding is performed.
        """
        y, sr = wav
        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=1024, n_mels=shape)
        mel = librosa.power_to_db(mel)
        mel = librosa.util.normalize(mel) + 1
        rows, cols = mel.shape
        diff = cols - shape
        if diff > 0:  # Random crop
            beginning_col = np.random.randint(diff)
            mel = mel[:, beginning_col:beginning_col + shape]
        elif diff < 0:  # Random zero-pad
            zeros = np.zeros((shape, shape), dtype=np.float32)
            beginning_col = np.random.randint(shape - cols)
            zeros[..., beginning_col:beginning_col + cols] = mel
            mel = zeros
        return mel

    def __getitem__(self, idx):
        path_to_wav = self.paths_to_wavs_list[idx]
        wav = self.read_audio(path_to_wav)
        if self.transform:
            wav = self.transform(wav)
        if self.spectrogram_type == 'mel':
            spectrogram = self.make_melspectrogram(wav, self.spectrogram_shape)
            spectrogram = np.expand_dims(spectrogram, axis=0)
        class_label = self.get_emotion_label(path_to_wav)
        return torch.from_numpy(spectrogram).float(), class_label
        # return spectrogram, class_label


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



def train_test_loaders(dataset, batch_size, random_seed=42, validation_split=0.2):
    """
    Create train and test DataLoaders
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=8)
    return train_loader, test_loader



if __name__ == '__main__':
    iemocap = IemocapDataset(path='datasets\\iemocap', name='IEMOCAP', label_type='four', spectrogram_shape=224)
    iemocap_train_loader, iemocap_test_loader = train_test_loaders(iemocap, batch_size=1)
    print(len(iemocap_train_loader.dataset))
    print(len(iemocap_test_loader.dataset))
    for i in range(len(iemocap)):
        spec, label = iemocap[i]
        print(spec.shape)

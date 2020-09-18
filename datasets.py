import os
import librosa
import torch 
import numpy as np

def get_paths_to_wavs(path_to_dataset):
    file_paths_list = []
    for root, _dirs, files in os.walk(path_to_dataset):  # Iterate over files in directory
        if len(files) != 0:
            file_paths_list += [os.path.join(root, f) for f in files if f.endswith('.wav')]
    return file_paths_list


class Denoiser:
    '''
    Denoise using spectral gating. Requires noise sample.
    '''
    def __call__(self, sample_wav, noise_wav):
        raise NotImplementedError


class Normalizer:
    '''
    Normalize amplitudes 
    '''
    def __call__(self, wav):
        raise NotImplementedError


class RemoveSilence:
    '''
    Remove silent parts 
    '''
    def __call__(self, wav):
        raise NotImplementedError


class TimeShift:
    def __call__(self, wav):
        raise NotImplementedError


class PitchShift:
    def __call__(self, wav):
        raise NotImplementedError




class RavdessDataset(torch.utils.data.Dataset):
    emotions_dict = {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
        }

    def __init__(self, path, transform=None):
        super(RavdessDataset, self).__init__()
        self.name = 'RAVDESS'
        self.path = path
        self.paths_to_wavs_list = get_paths_to_wavs(path)
        self.transform = transform

    def get_class_label(self, path_to_file):
        '''
        Parse the filename, which has the following pattern:
        modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        e.g., '02-01-06-01-02-01-12.wav'
        '''
        file_name = os.path.split(path_to_file)[1]
        file_name = file_name[:-4]
        class_label = int(file_name.split('-')[2]) - 1 # 2 is a number of emotion code
        return class_label

    def __len__(self):
        return len(self.paths_to_wavs_list)

    def read_audio(self, path_to_wav):
        '''
        Read .wav file using librosa, keeping orignal samplerate
        '''
        y, sr = librosa.load(path_to_wav, sr=None)
        return (y, sr)

    def random_crop(self, wav, length):
        '''
        Randomly crop audiofile to desired length (in samples)
        '''
        raise NotImplementedError

    def make_spectrogram(self, wav, shape):
        '''
        Ordinary spectrogram
        '''

    def make_melspectrogram(self, wav, shape):
        '''
        Mel-scaled spectrogram
        '''

    def __getitem__(self, idx):
        path_to_wav = self.paths_to_wavs_list[idx]
        wav = self.read_audio(path_to_wav)
        # !!!!APPLY PREPROCESSING HERE!!!!
        # !!!!RANDOMLY CROP              

        
        #mfccs = (mfccs - mfccs.mean())/np.std(mfccs)

        actual_mfcc_cols = mfccs.shape[1]

        # prmitive time-shifting augmentation
        target_real_diff = actual_mfcc_cols - self.mfcc_cols
        if target_real_diff > 0:
            
            if self.mode == 'TRAIN':
                beginning_col = np.random.randint(target_real_diff)
            else:
                beginning_col = actual_mfcc_cols//2 - self.mfcc_cols//2

            mfccs = mfccs[:, beginning_col:beginning_col + self.mfcc_cols]
            #mfccs = mfccs[:, beginning_col:beginning_col + self.mfcc_cols]

        elif target_real_diff < 0:
            zeros = np.zeros((self.mfcc_rows, self.mfcc_cols), dtype=np.float32)
            
            if self.mode == 'TRAIN':
                beginning_col = np.random.randint(self.mfcc_cols-actual_mfcc_cols)
            else:
            
                beginning_col = self.mfcc_cols//2 - actual_mfcc_cols//2
            zeros[..., beginning_col:beginning_col+actual_mfcc_cols] = mfccs
            #zeros[..., beginning_col:beginning_col+actual_mfcc_cols] = mfccs
            mfccs = zeros
            #mfccs = np.pad(mfccs, ((0, 0), (0, np.abs(target_real_diff))), constant_values=(0), mode='constant')

        # make the data compatible to pytorch 1-channel CNNs format
        mfccs = np.expand_dims(mfccs, axis=0)

        # normalize spectrogram
        mfccs = (mfccs - mfccs.mean())/np.std(mfccs)

        # Parse the filename, which has the following pattern:
        # modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
        # e.g., '02-01-06-01-02-01-12.wav'
        #file_name = os.path.split(path_to_wav)[1]
        #file_name = file_name[:-4]
        #class_label = int(file_name.split('-')[2]) - 1 # 2 is a number of emotion code
        #class_label = np.array(class_label)
        class_label = self.get_class_label(path_to_wav)
        # !!!!!!!!!
        return torch.from_numpy(mfccs), class_label#, path_to_wav



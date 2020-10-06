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


class IemocapDataset(torch.utils.data.Dataset):
    emotions_dict = {
        'ang': 0,
        'hap': 1,
        'neu': 2,
        'sad': 3,
        'exc': 4,
        'fru': 5,
    }

    speakers_dict = {
        'Ses01M': 0,
        'Ses01F': 1,
        'Ses02M': 2,
        'Ses02F': 3,
        'Ses03M': 4,
        'Ses03F': 5,
        'Ses04M': 6,
        'Ses04F': 7,
        'Ses05M': 8,
        'Ses05F': 9,
    }

    genders_dict = {
        'F': 0,
        'M': 1,
    }

    def __init__(self, pickle_path, wavs_path, egemaps_path, path_for_parser,
                 base_name, label_type='original', spectrogram_type='melspec', spectrogram_shape=128,
                 preprocessing=False, augmentation=False, padding='zero', mode='train', tasks=['emotion']):
        super(IemocapDataset, self).__init__()
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
#         labels = []
#         for task in self.tasks:
#             labels.append(file_instance[task])
        data = img
#         target = labels[0]
#         return data, self.emotions_dict[target]
        emotion = file_instance['emotion']
        speaker = file_instance['speaker']
        gender = file_instance['gender']
        return data, (self.emotions_dict[emotion], self.speakers_dict[speaker], self.genders_dict[gender])

                      
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






"""
This class parses the IEMOCAP DATASET and receives text,audio, and evaluation data.
"""


class IEMOCAP_Dataset_parser:
    datasetPath = "NULL"

    def __init__(self, Dataset_path):
        if os.path.exists(Dataset_path):
            self.datasetPath = Dataset_path
        else:
            assert os.path.lexists(Dataset_path), 'Error:  dataset ' \
                                                  'path does not exists!'

    """
    This method reads the text,audio and evaluation data of IEMOCAP dataset and creates a dictionary.
    Returns a dictionary with keys:
    main_key:  Utterance-Id
    partial_keys: Speaker-Id, Gender, Transcription, Wav, Freq, Emotion, Valence, Action, Dominance
    """

    def read_dataset(self):

        text_paths, wav_paths, eval_paths = self.gather_paths()

        text_dict = self.read_text_data(text_paths)

        eval_dict = self.read_eval_data(eval_paths)
        audio_dict = self.read_audio_data(wav_paths)

        final_dict = {}
        value_dict = {}
        for utt_id in eval_dict:
            value_dict = {"Speaker-Id": text_dict[utt_id]["Speaker-Id"],
                          "Gender": text_dict[utt_id]["Gender"],
                          "Transcription": text_dict[utt_id]["Transcription"],
                          "Wav": audio_dict[utt_id]["Wav"],
                          "Freq": audio_dict[utt_id]["Freq"],
                          "Emotion": eval_dict[utt_id]["Emotion"],
                          "Valence": eval_dict[utt_id]["Valence"],
                          "Action": eval_dict[utt_id]["Action"],
                          "Dominance": eval_dict[utt_id]["Dominance"]}
            final_dict[utt_id] = value_dict

        return final_dict

    """
    This method collects all files paths containing transcriptions , wavs and evaluations.
    Returns a 3 lists containing those paths.
    1st list transcription paths.
    2nd list wav paths.
    3rd list eval paths.
    """

    def gather_paths(self):
        transcription_paths = []
        wav_paths = []
        eval_paths = []

        listofSessions = ["Session1", "Session2", "Session3",
                          "Session4", "Session5"]

        for Session in listofSessions:
            path = os.path.join(self.datasetPath, Session, "dialog", "transcriptions/")
            p = os.listdir(path)
            transcription_paths = transcription_paths + glob.glob(path + "*.txt")

            path = os.path.join(self.datasetPath, Session, "dialog", "EmoEvaluation/")
            eval_paths = eval_paths + glob.glob(path + "*.txt")

            path = os.path.join(self.datasetPath, Session, "sentences", "wav/")
            wav_paths = wav_paths + glob.glob(path + "**/*.wav", recursive=True)

        return transcription_paths, wav_paths, eval_paths

    """
    This method takes all transcription paths and returns a nested dictionary with keys:
    main_key:Utterance-Id
    partial_keys: Speaker-Id,  Gender, Transcription
    """

    def read_text_data(self, text_paths):
        all_Utterances = []
        all_Genders = []
        all_Transcriptions = []
        all_Speakers = []

        for text_file in text_paths:
            utt_list, gender_list, trans_list = self.read_trans_from_file(text_file)
            all_Utterances = utt_list + all_Utterances
            all_Genders = gender_list + all_Genders
            all_Transcriptions = trans_list + all_Transcriptions
            speaker_id = utt_list[0].split("_")[0]
            speakers_list = [speaker_id] * len(utt_list)
            all_Speakers = all_Speakers + speakers_list

        # create text_dict
        text_dict = {}
        value_dict = {}
        i = 0
        for utt_id in all_Utterances:
            value_dict = {"Gender": all_Genders[i], "Speaker-Id": all_Speakers[i],
                          "Transcription": all_Transcriptions[i]}
            text_dict[utt_id] = value_dict
            i += 1

        return text_dict

    """
    This method takes as input a file(containing utt and trans) and returns 3 lists.
    The 1st list contains every utterance key.
    The 2nd list contains every gender.
    The 3rd list contains every transcription.
    The lists are sorted so as each utterance key has its relevant transcription at the same index.
    """

    def read_trans_from_file(self, text_file):

        Utterance_id_list = []
        Gender_list = []
        Transcription_list = []

        infile = open(text_file, "r")
        for line in infile.readlines():
            (utterance_key, transcription) = line.split(":")
            utterance_key = line.split(" ")[0]
            # utterance_key must start with Ses and  not
            # contains MX.. or FX..
            if (utterance_key.startswith("Ses")
                    and ("MX" not in utterance_key)
                    and ("FX" not in utterance_key)):
                # delete newline on transcription end  and
                # also delete space at transcription's
                # start
                transcription = transcription[1:len(
                    transcription) - 1]
                gender = utterance_key[len(
                    utterance_key) - 4:len(utterance_key) - 3]

                Utterance_id_list.append(utterance_key)
                Gender_list.append(gender)
                Transcription_list.append(transcription)

        return Utterance_id_list, Gender_list, Transcription_list

    """
    This method takes all text files paths containing the emotion evaluation and returns a nested dictionary with keys:
    main_key: Utterance-Id
    partial_keys: Emotion,Valence,Action,Dominance.
    """

    def read_eval_data(self, eval_paths):
        Utt_id_list = []
        Emotion_list = []
        Valence_list = []
        Action_list = []
        Dominance_list = []

        for eval_file in eval_paths:
            utt_list, em_list, val_list, ac_list, dom_list = self.read_aff_from_file(eval_file)
            Utt_id_list = Utt_id_list + utt_list
            Emotion_list = Emotion_list + em_list
            Valence_list = Valence_list + val_list
            Action_list = Action_list + ac_list
            Dominance_list = Dominance_list + dom_list

        # create eval_dict
        eval_dict = {}
        value_dict = {}
        i = 0
        for utt_id in Utt_id_list:
            value_dict = {"Emotion": Emotion_list[i], "Valence": Valence_list[i], "Action": Action_list[i],
                          "Dominance": Dominance_list[i]}
            eval_dict[utt_id] = value_dict
            i += 1

        print(len(Utt_id_list))
        return eval_dict

    """
    This method takes as input  the evaluation file.
    It also takes a list with utterance-Ids of the relevant
    """

    def read_aff_from_file(self, eval_file):
        Utterance_id_list = []
        Emotion_list = []
        Valence_list = []
        Action_list = []
        Dominance_list = []

        infile = open(eval_file, "r")
        for line in infile.readlines():
            if (("Ses" in line) and ("MX" not in line) and ("FX" not in line)):
                line_splitted = line.split("\t")
                utt_id = line_splitted[1]
                emotion = line_splitted[2]
                if (not emotion == "xxx"):
                    split = line_splitted[3].split(",")
                    valence = split[0]
                    valence = valence[1:len(valence)]
                    action = split[1]
                    action = action[1:len(action)]
                    dominance = split[2]
                    dominance = dominance[1:len(dominance) - 2]
                    Utterance_id_list.append(utt_id)
                    Emotion_list.append(emotion)
                    Valence_list.append(valence)
                    Action_list.append(action)
                    Dominance_list.append(dominance)

        return Utterance_id_list, Emotion_list, Valence_list, Action_list, Dominance_list

    """
    This method takes as input all wav files paths and returns a nested dictionary with keys:
    main_key: Utterance-Id
    partial_keys: Wav,Freq
    """

    def read_audio_data(self, audio_paths):
        Utt_id_list = []
        Wav_list = []
        Freq_list = []

        for wav in audio_paths:
            utt_id = wav.split("\\")[9]
            utt_id = utt_id[0:len(utt_id) - 4]
            (wav_array, freq) = self.dummy_audio_reader(wav)
            Utt_id_list.append(utt_id)
            Wav_list.append(wav_array)
            Freq_list.append(freq)

        # create audio_dict:
        audio_dict = {}
        value_dict = {}
        i = 0
        for utt_id in Utt_id_list:
            value_dict = {"Wav": Wav_list[i], "Freq": Freq_list[i]}
            audio_dict[utt_id] = value_dict
            i += 1
        return audio_dict

    """
    This is a dummy audio reader!
    The original should take a wavfile and return normalized np-array of wav and the relevant freq!
    """

    def dummy_audio_reader(self, infile):
        import numpy as np
        return [5, 4, 1, 2], 260000


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
        noise_reduced = nr.reduce_noise(audio_clip=y, noise_clip=self.noise, prop_decrease=0.9, verbose=False,
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
    

    

def train_test_loaders(dataset, validation_ratio=0.2, **kwargs):
    """
    Create train and test DataLoaders
    :param kwargs: keyword arguments for DataLoader
    :return: train and test loaders
    """
    dataset_size = len(dataset)
    test_size = int(np.floor(validation_ratio * dataset_size))
    train_size = dataset_size - test_size
    print('TRAIN SIZE {}'.format(train_size))
    print('TEST SIZE {}'.format(test_size))
    train_dataset, test_dataset = random_split(dataset, (train_size, test_size),
                                               generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    iemocap_four_noprep_train = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                               egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                               path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                               base_name='IEMOCAP-4', label_type='four', preprocessing=False,
                                               mode='train')
    iemocap_four_noprep_test = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                              egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                              path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                              base_name='IEMOCAP-4', label_type='four', preprocessing=False,
                                              mode='test')
    iemocap_four_prep_train = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                             egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                             path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                             base_name='IEMOCAP-4', label_type='four', preprocessing=True, mode='train')
    iemocap_four_prep_test = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                            egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                            path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                            base_name='IEMOCAP-4', label_type='four', preprocessing=True, mode='test')
    iemocap_original_noprep_train = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                                   egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                                   path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                                   base_name='IEMOCAP', label_type='original', preprocessing=False,
                                                   mode='train')
    iemocap_original_noprep_test = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                                  egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                                  path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                                  base_name='IEMOCAP', label_type='original', preprocessing=False,
                                                  mode='test')
    iemocap_original_prep_train = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                                 egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                                 path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                                 base_name='IEMOCAP', label_type='original', preprocessing=True,
                                                 mode='train')
    iemocap_original_prep_test = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                                egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                                path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                                base_name='IEMOCAP', label_type='original', preprocessing=True,
                                                mode='test')

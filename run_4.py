from datasets import *


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
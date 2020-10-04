from iemocap import *




# 64 4 cats

iemocap_four_noprep_train = IemocapDataset(  # Без препроцессинга, тренировочный
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='train', preprocessing=False,
    augmentation=False, padding='repeat', spectrogram_shape=64, spectrogram_type='melspec', tasks=['emotion']
)
iemocap_four_noprep_test = IemocapDataset(  # Без препроцессинга, тестовый
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='test', preprocessing=False,
    augmentation=False, padding='repeat', spectrogram_shape=64, spectrogram_type='melspec', tasks=['emotion']
)
iemocap_four_prep_train = IemocapDataset(  # C препроцессингом, тренировочный
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='train', preprocessing=True,
    augmentation=False, padding='repeat', spectrogram_shape=64, spectrogram_type='melspec', tasks=['emotion']
)
iemocap_four_prep_test = IemocapDataset(  # С препроцессингом, тестовый
    PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
    base_name='IEMOCAP-4', label_type='four', mode='test', preprocessing=True,
    augmentation=False, padding='repeat', spectrogram_shape=64, spectrogram_type='melspec', tasks=['emotion']
)
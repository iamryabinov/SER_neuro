from datasets import IemocapDataset
from constants import *

iemocap_256_noprep = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                    egemaps_path=IEMOCAP_PATH_TO_EGEMAPS, path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                    base_name='IEMOCAP', label_type='four', spectrogram_shape=256,
                                    preprocessing=False, augmentation=False)
iemocap_256_prep = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                  egemaps_path=IEMOCAP_PATH_TO_EGEMAPS, path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                                  base_name='IEMOCAP', label_type='four', spectrogram_shape=256,
                                  preprocessing=True, augmentation=False)

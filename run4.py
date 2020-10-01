from datasets import IemocapDataset
from constants import *

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
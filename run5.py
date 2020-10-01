from datasets import IemocapDataset
from constants import *
from PIL import Image

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
#
# for i in range(len(iemocap_four_noprep_train)):
#     if i % 1500 == 0:
#         for j in range(5):
#             img, _ = iemocap_four_noprep_train[i]
#             img = Image.fromarray(img)
#             img.show()

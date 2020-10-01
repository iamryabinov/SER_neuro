from datasets import IemocapDataset
from constants import *
import matplotlib.pyplot as plt

iemocap_four_noprep_train = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path=IEMOCAP_PATH_TO_WAVS,
                                           egemaps_path=IEMOCAP_PATH_TO_EGEMAPS,
                                           path_for_parser=IEMOCAP_PATH_FOR_PARSER, spectrogram_shape=256,
                                           base_name='IEMOCAP-4', label_type='four', preprocessing=False,
                                           mode='train')
for i in range(len(iemocap_four_noprep_train)):
    if i == 6:
        iemocap_four_noprep_train.spectrogram_shape = 256
        iemocap_four_noprep_train.augmentation = True
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
        iemocap_four_noprep_train.mode = 'test'
        iemocap_four_noprep_train.padding = 'repeat'
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
        ax = iemocap_four_noprep_train.show_image(i)
        plt.show()
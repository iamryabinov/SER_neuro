from datasets import *
from constants import *
import PIL

dataset1 = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path='E:\\Projects\\SER\\new\\datasets\\',
                          egemaps_path=IEMOCAP_PATH_TO_EGEMAPS, path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                          base_name='IEMOCAP', label_type='original', spectrogram_shape=256,
                          preprocessing=False, augmentation=False)

dataset2 = IemocapDataset(pickle_path=PATH_TO_PICKLE, wavs_path='E:\\Projects\\SER\\new\\datasets\\',
                         egemaps_path=IEMOCAP_PATH_TO_EGEMAPS, path_for_parser=IEMOCAP_PATH_FOR_PARSER,
                         base_name='IEMOCAP', label_type='original', spectrogram_shape=256,
                         preprocessing=True, augmentation=False)

for i in range(len(dataset2)):
    img, label = dataset1[i]
    img = Image.fromarray(img)
    img.show()
    img, label = dataset2[i]
    img = Image.fromarray(img)
    img.show()

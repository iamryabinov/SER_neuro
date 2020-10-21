from iemocap import *
from multitask_training_session import *
from models_multi_task import *

if __name__ == '__main__':
    iemocap_224_noprep_train = IemocapDataset(  # Без препроцессинга, тренировочный
        PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
        base_name='IEMOCAP-4', label_type='four', mode='train', preprocessing=False,
        augmentation=False, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec',
        tasks=('emotion', 'speaker', 'gender')
    )

    iemocap_224_noprep_test = IemocapDataset(  # Без препроцессинга, тестовый
        PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
        base_name='IEMOCAP-4', label_type='four', mode='test', preprocessing=False,
        augmentation=False, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec',
        tasks=('emotion', 'speaker', 'gender')
    )
    print(len(iemocap_224_noprep_train) + len(iemocap_224_noprep_test))


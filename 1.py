from ramas import *
import pickle




# 64 4 cats

train_dataset = RamasDataset(
    wavs_path=RAMAS_PATH_TO_WAVS_DESCRETE, base_name='RAMAS_descrete',  # Тренировочный
    spectrogram_shape=224,
    augmentation=False, padding='zero', mode='train', tasks='emotion',
    emotions=('Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad emotion', 'Scared', 'Shame', 'Surprised', 'Tiredness')
)
test_dataset = RamasDataset(
    wavs_path=RAMAS_PATH_TO_WAVS_DESCRETE, base_name='RAMAS_descrete',  # Тестовый
    spectrogram_shape=224,
    augmentation=False, padding='zero', mode='test', tasks='emotion',
    emotions=('Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad emotion', 'Scared', 'Shame', 'Surprised', 'Tiredness')
)

broken_paths = []
for i in range(len(train_dataset)):
    try:
        spec, labels = train_dataset[i]
    except ValueError:
        path_to_broken_file = train_dataset.paths_to_wavs[i]
        broken_paths.append(path_to_broken_file)
        print('----------------------------------------')
        print('!!!{} was broken, appended to list... Found total {} broken files so far...'
              .format(path_to_broken_file, len(broken_paths)))
        print('----------------------------------------')

for i in range(len(test_dataset)):
    try:
        spec, labels = test_dataset[i]
    except ValueError:
        path_to_broken_file = test_dataset.paths_to_wavs[i]
        broken_paths.append(path_to_broken_file)
        print('----------------------------------------')
        print('!!!{} was broken, appended to list... Found total {} broken files so far...'
              .format(path_to_broken_file, len(broken_paths)))
        print('----------------------------------------')

print('Found total {} broken files!'.format(len(broken_paths)))
with open('broken_files.pkl', 'wb') as f:
    pickle.dump(broken_paths, f)

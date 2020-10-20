from datasets.ramas import *
import pickle



if __name__ == '__main__':
    # 64 4 cats

    train_dataset_descrete = RamasDataset(
        wavs_path=RAMAS_PATH_TO_WAVS_DESCRETE, base_name='RAMAS_descrete',  # Тренировочный
        spectrogram_shape=224,
        augmentation=False, padding='zero', mode='train', tasks='emotion',
        emotions=('Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad emotion', 'Scared', 'Shame', 'Surprised', 'Tiredness')
    )
    test_dataset_descrete = RamasDataset(
        wavs_path=RAMAS_PATH_TO_WAVS_DESCRETE, base_name='RAMAS_descrete',  # Тестовый
        spectrogram_shape=224,
        augmentation=False, padding='zero', mode='test', tasks='emotion',
        emotions=('Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad emotion', 'Scared', 'Shame', 'Surprised', 'Tiredness')
    )

    train_dataset_dom_sub = RamasDataset(
        wavs_path=RAMAS_PATH_TO_WAVS_DOM_SUB, base_name='RAMAS_dom_sub',  # Тренировочный
        spectrogram_shape=224,
        augmentation=False, padding='zero', mode='train', tasks='emotion',
        emotions=('Domination', 'Submission')
    )
    test_dataset_dom_sub = RamasDataset(
        wavs_path=RAMAS_PATH_TO_WAVS_DOM_SUB, base_name='RAMAS_dom_sub',  # Тестовый
        spectrogram_shape=224,
        augmentation=False, padding='zero', mode='test', tasks='emotion',
        emotions=('Domination', 'Submission')
    )

    broken_paths = []
    for i in range(len(train_dataset_descrete)):
        try:
            spec, labels = train_dataset_descrete[i]
        except ValueError:
            path_to_broken_file = train_dataset_descrete.paths_to_wavs[i]
            broken_paths.append(path_to_broken_file)
            print('----------------------------------------')
            print('!!!{} was broken, appended to list... Found total {} broken files so far...'
                  .format(path_to_broken_file, len(broken_paths)))
            print('----------------------------------------')
    print('Found total {} broken files!'.format(len(broken_paths)))
    with open('broken_files.pkl', 'wb') as f:
        pickle.dump(broken_paths, f)

    for i in range(len(test_dataset_descrete)):
        try:
            spec, labels = test_dataset_descrete[i]
        except ValueError:
            path_to_broken_file = test_dataset_descrete.paths_to_wavs[i]
            broken_paths.append(path_to_broken_file)
            print('----------------------------------------')
            print('!!!{} was broken, appended to list... Found total {} broken files so far...'
                  .format(path_to_broken_file, len(broken_paths)))
            print('----------------------------------------')
    print('Found total {} broken files!'.format(len(broken_paths)))
    with open('broken_files.pkl', 'wb') as f:
        pickle.dump(broken_paths, f)

    for i in range(len(train_dataset_dom_sub)):
        try:
            spec, labels = train_dataset_dom_sub[i]
        except ValueError:
            path_to_broken_file = train_dataset_dom_sub.paths_to_wavs[i]
            broken_paths.append(path_to_broken_file)
            print('----------------------------------------')
            print('!!!{} was broken, appended to list... Found total {} broken files so far...'
                  .format(path_to_broken_file, len(broken_paths)))
            print('----------------------------------------')
    print('Found total {} broken files!'.format(len(broken_paths)))
    with open('broken_files.pkl', 'wb') as f:
        pickle.dump(broken_paths, f)

    for i in range(len(test_dataset_dom_sub)):
        try:
            spec, labels = test_dataset_dom_sub[i]
        except ValueError:
            path_to_broken_file = test_dataset_dom_sub.paths_to_wavs[i]
            broken_paths.append(path_to_broken_file)
            print('----------------------------------------')
            print('!!!{} was broken, appended to list... Found total {} broken files so far...'
                  .format(path_to_broken_file, len(broken_paths)))
            print('----------------------------------------')
    print('Found total {} broken files!'.format(len(broken_paths)))
    with open('broken_files.pkl', 'wb') as f:
        pickle.dump(broken_paths, f)

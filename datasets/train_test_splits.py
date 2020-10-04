import sklearn.model_selection as ms
import os
from constants import *
import shutil


def get_paths_to_wavs(path_to_dataset_wavs):
    file_paths_list = []
    noise_file_path = ''
    for root, _dirs, files in os.walk(path_to_dataset_wavs):  # Iterate over files in directory
        if len(files) != 0:
            for f in files:
                if f.endswith('.wav'):
                    if 'noise' in f:
                        noise_file_path = os.path.join(root, f)
                    else:
                        file_paths_list.append(os.path.join(root, f))
                else:
                    continue
    if file_paths_list == []:
        raise FileNotFoundError('Returned empty list!')
    return file_paths_list, noise_file_path


def get_emotion_label(file_path):
    """
    Parse the filename, return emotion label
    """
    file_name = os.path.split(file_path)[1]
    file_name = file_name[:-4]
    emotion_name = file_name.split('_')[-1]  # the last is a position of emotion code
    return emotion_name


def make_X_y_dict(file_paths_list):
    y = []
    dictionary = {
        'X': file_paths_list,
        'y': y
    }
    for file_path in file_paths_list:
        y.append(get_emotion_label(file_path))
    return dictionary


def make_train_test_folders(wavs_folder, train_folder_path, test_folder_path, seed=42):
    if not os.path.exists(train_folder_path):
        os.mkdir(train_folder_path)
    if not os.path.exists(test_folder_path):
        os.mkdir(test_folder_path)
    files = os.listdir(train_folder_path) + os.listdir(test_folder_path)
    for file in files:
        os.remove(file)
    dict = make_X_y_dict(get_paths_to_wavs(wavs_folder)[0])
    X = dict['X']
    y = dict['y']
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)
    for file, i in zip(X_train, range(len(X_train))):
        print("{}/{}".format(i+1, len(X_train)))
        shutil.copy2(file, train_folder_path)
    for file, i in zip(X_test, range(len(X_test))):
        print("{}/{}".format(i + 1, len(X_test)))
        shutil.copy2(file, test_folder_path)


if __name__ == '__main__':
    make_train_test_folders(RAMAS_PATH_TO_WAVS, RAMAS_PATH_TO_WAVS + 'train\\', RAMAS_PATH_TO_WAVS + 'test\\')
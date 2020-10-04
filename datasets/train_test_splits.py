import sklearn.model_selection as ms
import os
from constants import *
import shutil
import tqdm
import pandas as pd


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


def copy_domination_submission_files(in_path, out_path):
    for file in os.listdir(in_path):
        if file.endswith('Domination.wav') or file.endswith('Submission.wav'):
            shutil.copy2(os.path.join(in_path, file), out_path)
            print('Copied {}'.format(file))

def cut_ramas_files():
    # !!!!!!!!!!!
    # Важно прописать путь windows именно в линуксовском стиле через /
    # иначе возникают проблемы с косыми чертами при передаче этого пути в командную строку
    source_path = 'E:/Projects/SER/datasets/RAMAS/Audio/'
    video_list = [item for item in os.listdir(source_path) if item.endswith('.wav')]

    # !!!!!!!!!!!
    # Важно прописать путь windows именно в линуксовском стиле через /
    # иначе возникают проблемы с косыми чертами при передаче этого пути в командную строку
    target_path = 'E:/Projects/SER/datasets/RAMAS/Audio/Audio_cut/'
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    # !!!!!!!!!!!
    # Здесь можно оставить виндовский стиль, т.к. этот путь не подается в комендную строку
    path_to_csvs = 'E:\\Projects\\SER\\datasets\\RAMAS\\Annotations_by_emotions'
    csvs_list = [item for item in os.listdir(path_to_csvs) if (item.endswith('.csv') and 'labeled' not in item)]

    for csv_name in csvs_list:
        path_to_csv = os.path.join(path_to_csvs, csv_name)
        df = pd.read_csv(path_to_csv, delimiter=';')
        class_name = csv_name[:-4].split('_')[1]

        # iterate over rows of the dataframe
        for idx, row in tqdm(df.iterrows()):
            i = 1
            ID = row['ID']
            in_file_mane = row['File']
            start_time = row['Start']
            end_time = row['End']
            class_name = row['emotion']
            in_file_path = source_path + in_file_mane + '_mic.wav'
            out_file_name = in_file_mane + '_{}_{}.wav'.format(idx, class_name)
            out_file_path = target_path + out_file_name
            ffmpeg_str = 'ffmpeg -ss {} -i {} -to {} -c copy {}'.format(start_time, in_file_path, end_time,
                                                                        out_file_path)
            i += 1
            os.system(ffmpeg_str)

def segment_ramas_files():
    source_path = 'E:/Projects/SER/datasets/RAMAS/Audio/Audio_cut/domination_submission/'
    audio_list = [item for item in os.listdir(source_path) if item.endswith('.wav')]
    target_path = 'E:/Projects/SER/datasets/RAMAS/Audio/Audio_cut/domination_submission/segmented/'
    for in_file in audio_list:
        in_file_path = os.path.join(source_path, in_file)
        print('==============================================')
        print(in_file_path)
        ffmpeg_string = 'ffmpeg -i {} -f segment -segment_time 2 -c copy {}%03d_{}'.format(
            in_file_path, target_path, in_file
        )
        os.system(ffmpeg_string)

if __name__ == '__main__':
    segment_ramas_files()
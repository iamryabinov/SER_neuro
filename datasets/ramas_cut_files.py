import os
import pandas as pd
import cv2

from tqdm import tqdm

if __name__ == '__main__':
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

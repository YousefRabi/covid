from pathlib import Path
import glob
import shutil
import ast

import pandas as pd

from utils.utils import get_train_test_image_sizes


def copy_images_for_yolo(input_path: Path,
                         output_path: Path):
    files = list(input_path.rglob('*.png'))
    output_path.mkdir(parents=True, exist_ok=True)

    for f in files:
        shutil.copy(f, output_path / f.name)


def create_labels_for_yolo(split_file, labels_output_path: Path):
    labels_output_path.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(split_file)
    sizes = get_train_test_image_sizes()

    groupby = s.groupby('id')
    for image_id, group in groupby:
        image_id = image_id.split('_')[0]
        print(image_id, len(group))
        out_file_path = labels_output_path / f'{image_id}.txt'
        out = open(out_file_path, 'w')
        for _, row in group.iterrows():
            if str(row['boxes']) == 'nan':
                continue
            boxes = ast.literal_eval(row['boxes'])
            for box in boxes:
                xmin, ymin = box['x'], box['y']
                width, height = box['width'], box['height']
                xmin = xmin / sizes[image_id][1]
                ymin = ymin / sizes[image_id][0]
                xmax = xmin + width / sizes[image_id][1]
                ymax = ymin + height / sizes[image_id][0]
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin
                out.write('{} {} {} {} {}\n'.format('0', center_x, center_y, width, height))
        out.close()


def create_split_for_yolo5(split_file, out_folder: Path):
    training_img_directory = out_folder / 'train' / 'images'
    training_img_directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(split_file)
    folds = data['fold'].max() + 1

    for fold_num in range(folds):
        part = data[data['fold'] != fold_num].copy()
        train_ids = part['id'].values
        train_df = data[data['id'].isin(train_ids)]
        out_path_train = out_folder / f'fold_{fold_num}_train.txt'
        train_df['id'] = training_img_directory.as_posix() + '/' + train_df['id'] + '.png'
        out = open(out_path_train, 'w')
        for id in train_df['id'].values:
            out.write('{}\n'.format(id))
        out.close()

        part = data[data['fold'] == fold_num].copy()
        valid_ids = part['id'].values
        valid_df = data[data['id'].isin(valid_ids)]
        out_path_valid = out_folder / f'fold_{fold_num}_valid.csv'
        valid_df['id'] = training_img_directory.as_posix() + '/' + valid_df['id'] + '.png'
        out = open(out_path_valid, 'w')
        for id in valid_df['id'].values:
            out.write('{}\n'.format(id))
        out.close()

        # Create XML
        out = open(out_folder / 'fold_{}.xml'.format(fold_num), 'w')
        out.write('train: {}\n'.format(out_path_train))
        out.write('val: {}\n'.format(out_path_valid))
        out.write('nc: {}\n'.format(1))
        out.write('names: {}\n'.format(['opacity']))
        out.close()


def create_fold_folders(split_file: str, parent_path: Path, fold_num):
    files = list(parent_path.rglob('*.png'))
    out_fold_folder = parent_path.parent / 'images' / 'train_fold{}'.format(fold_num)
    out_fold_folder.mkdir(parents=True, exist_ok=True)

    print(len(files))
    s = pd.read_csv(split_file)
    part = s[s['fold'] == fold_num].copy()
    fold_ids = set(part['id'].values)
    print(len(fold_ids))
    for f in files:
        file_id = f.stem
        if file_id in fold_ids:
            shutil.copy(f, out_fold_folder + file_id + '.png')


def main():
    split_file = 'data/processed/stratified_kfold_study_split_5_42.csv'
    out_folder = Path('data/processed/yolov5_data')
    out_folder.mkdir(parents=True, exist_ok=True)

    copy_images_for_yolo(Path('data/processed/train'),
                         Path(out_folder / 'train' / 'all_images'))
    copy_images_for_yolo(Path('data/processed/test'),
                         Path(out_folder / 'test' / 'images'))

    create_labels_for_yolo(split_file, out_folder / 'labels')
    create_split_for_yolo5(split_file, out_folder / 'train' / 'images')

    for i in range(5):
        create_fold_folders(split_file, out_folder / 'train' / 'all_images', i)


if __name__ == '__main__':
    main()

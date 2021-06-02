from pathlib import Path
import glob
import shutil
import ast

import pandas as pd
import numpy as np

from utils.utils import get_train_test_image_sizes


def copy_images_for_yolo(split_file,
                         input_path: Path,
                         output_path: Path):
    split_df = pd.read_csv(split_file)
    files = list(input_path.rglob('*.png'))
    output_path.mkdir(parents=True, exist_ok=True)

    for f in files:
        if str(split_df[split_df['id'] == f.stem + '_image']['boxes']) != 'nan':
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
        for _, row in group.iterrows():
            if str(row['boxes']) == 'nan':
                continue
            out = open(out_file_path, 'w')
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


def create_yolo_folds(split_file: str, imgs_folder: Path, labels_folder: Path, fold_num: int):
    split_df = pd.read_csv(split_file)
    train_df = split_df[split_df['fold'] != fold_num]
    val_df = split_df[split_df['fold'] == fold_num]
    train_ids = [image_id.split('_')[0] for image_id in train_df.id.values]
    val_ids = [image_id.split('_')[0] for image_id in val_df.id.values]

    image_names = np.array([img_path.stem for img_path in imgs_folder.rglob('*.png')])

    train_image_names = [image_name for image_name in image_names if image_name in train_ids]
    valid_image_names = [image_name for image_name in image_names if image_name in val_ids]

    train_fold_folder = (imgs_folder.parent / f'train_fold_{fold_num}')
    (train_fold_folder / 'images').mkdir(parents=True, exist_ok=True)
    (train_fold_folder / 'labels').mkdir(parents=True, exist_ok=True)

    valid_fold_folder = (imgs_folder.parent / f'valid_fold_{fold_num}')
    (valid_fold_folder / 'images').mkdir(parents=True, exist_ok=True)
    (valid_fold_folder / 'labels').mkdir(parents=True, exist_ok=True)

    images_copied = 0
    for train_image_name in train_image_names:
        train_label_name = Path(train_image_name).with_suffix('.txt')
        try:
            shutil.copy(labels_folder / train_label_name,
                        train_fold_folder / 'labels' / train_label_name)
            shutil.copy((imgs_folder / train_image_name).with_suffix('.png'),
                        (train_fold_folder / 'images' / train_image_name).with_suffix('.png'))
            images_copied += 1
        except Exception as e:
            print(f'Exception: {e} -- Image name: {train_image_name}')
            continue

    print(f'Copied {images_copied} training images for fold {fold_num} to {train_fold_folder / "images"}')

    images_copied = 0
    for valid_image_name in valid_image_names:
        valid_label_name = Path(valid_image_name).with_suffix('.txt')
        try:
            shutil.copy(labels_folder / valid_label_name,
                        valid_fold_folder / 'labels' / valid_label_name)
            shutil.copy((imgs_folder / valid_image_name).with_suffix('.png'),
                        (valid_fold_folder / 'images' / valid_image_name).with_suffix('.png'))
            images_copied += 1
        except Exception as e:
            print(f'Exception: {e} -- Image name: {valid_image_name}')
            continue

    print(f'Copied {images_copied} validation images for fold {fold_num} to {valid_fold_folder / "images"}')


def main():
    split_file = 'data/processed/stratified_kfold_study_split_5_42.csv'
    out_folder = Path('data/processed/yolov5_data')
    out_folder.mkdir(parents=True, exist_ok=True)

    copy_images_for_yolo(split_file,
                         Path('data/processed/train'),
                         Path(out_folder / 'train' / 'all_images'))
    copy_images_for_yolo(split_file,
                         Path('data/processed/test'),
                         Path(out_folder / 'test' / 'images'))

    create_labels_for_yolo(split_file, out_folder / 'train' / 'all_labels')

    for i in range(5):
        create_yolo_folds(split_file,
                          out_folder / 'train' / 'all_images',
                          out_folder / 'train' / 'all_labels', i)


if __name__ == '__main__':
    main()

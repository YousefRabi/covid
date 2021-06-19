from pathlib import Path
import shutil

from sklearn.model_selection import KFold

import numpy as np


def copy_images_for_yolo(txt_labels_folder: Path,
                         images_folder: Path,
                         output_path: Path):
    image_ids = [label_path.stem for label_path in txt_labels_folder.rglob('*.txt')]
    output_path.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        image_name = Path(image_id).with_suffix('.png')
        shutil.copy(images_folder / image_name, output_path / image_name)


def create_yolo_folds(imgs_folder: Path, labels_folder: Path):
    image_names = np.array([img_path.stem for img_path in imgs_folder.rglob('*.png')])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_idx, valid_idx) in enumerate(kf.split(image_names)):

        train_image_names = image_names[train_idx]
        valid_image_names = image_names[valid_idx]

        fold_num = i

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
    txt_labels_folder = Path('data/processed/one_lung_dataset_yolo_labels')
    out_folder = Path('data/processed/one_lung_detector_yolov5_data')
    out_folder.mkdir(parents=True, exist_ok=True)

    copy_images_for_yolo(txt_labels_folder,
                         Path('data/processed/train/png_div_2'),
                         Path(out_folder / 'train' / 'all_images'))

    create_yolo_folds(out_folder / 'train' / 'all_images',
                      txt_labels_folder)


if __name__ == '__main__':
    main()

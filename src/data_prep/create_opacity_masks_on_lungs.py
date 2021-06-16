from pathlib import Path
from argparse import ArgumentParser
import ast
from tqdm import tqdm

import pandas as pd
import numpy as np

import skimage.io


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--original_images_dir')
    parser.add_argument('--lung_images_dir')
    parser.add_argument('--lung_labels_dir')
    parser.add_argument('--image_level_df')
    parser.add_argument('--opacity_mask_output_dir')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    original_images_dir = Path(args.original_images_dir)
    lung_images_dir = Path(args.lung_images_dir)
    lung_labels_dir = Path(args.lung_labels_dir)
    opacity_masks_dir = Path(args.opacity_mask_output_dir)
    opacity_masks_dir.mkdir(parents=True, exist_ok=True)

    image_df = pd.read_csv(args.image_level_df)

    for idx, row in tqdm(image_df.iterrows(), total=len(image_df)):
        image_id = row['id'].split('_')[0]

        original_image = skimage.io.imread(original_images_dir / (image_id + '.png'))
        lung_image = skimage.io.imread(lung_images_dir / (image_id + '.jpg'))

        if str(row['boxes']) == 'nan':
            opacity_mask = np.zeros(lung_image.shape[:2], dtype=np.uint8)
            skimage.io.imsave(opacity_masks_dir / (image_id + '.png'), opacity_mask)
            continue

        lung_yolo_bbox = read_lung_label(lung_labels_dir / (image_id + '.txt'))
        lung_abs_bbox = convert_yolo_to_abs(lung_yolo_bbox, original_image)

        opacity_original_bboxes = read_opacity_label(row['boxes'])
        opacity_lung_bboxes = convert_opacity_from_image_to_lung(opacity_original_bboxes, lung_abs_bbox)

        opacity_mask = make_mask(lung_image, opacity_lung_bboxes)

        skimage.io.imsave(opacity_masks_dir / (image_id + '.png'), opacity_mask)


def make_mask(image, opacity_bboxes_on_lungs):
    mask = np.zeros(image.shape, dtype=np.uint8)

    for opacity_bbox in opacity_bboxes_on_lungs:
        xmin, ymin, xmax, ymax = opacity_bbox

        mask[ymin:ymax, xmin:xmax] = 255

    return mask


def convert_opacity_from_image_to_lung(opacity_bboxes, lung_abs_bbox):
    opacity_bboxes_on_lungs = []

    for opacity_bbox in opacity_bboxes:
        xmin_on_lungs = max(0, opacity_bbox[0] - lung_abs_bbox[0])
        ymin_on_lungs = max(0, opacity_bbox[1] - lung_abs_bbox[1])

        opacity_width = opacity_bbox[2] - opacity_bbox[0]
        opacity_height = opacity_bbox[3] - opacity_bbox[1]

        xmax_on_lungs = min(lung_abs_bbox[2], xmin_on_lungs + opacity_width)
        ymax_on_lungs = min(lung_abs_bbox[3], ymin_on_lungs + opacity_height)

        opacity_bboxes_on_lungs.append([int(xmin_on_lungs),
                                        int(ymin_on_lungs),
                                        int(xmax_on_lungs),
                                        int(ymax_on_lungs)])

    return opacity_bboxes_on_lungs


def read_opacity_label(label_str):
    opacity_bboxes_on_image = ast.literal_eval(label_str)
    opacity_bboxes = [
        [bbox_dict['x'] / 2,
         bbox_dict['y'] / 2,
         (bbox_dict['x'] + bbox_dict['width']) / 2,
         (bbox_dict['y'] + bbox_dict['height']) / 2]
        for bbox_dict in opacity_bboxes_on_image
    ]
    return opacity_bboxes


def read_lung_label(txt_path):
    with open(txt_path) as f:
        lung_rel_bbox = f.read().splitlines()[0].split()
        lung_rel_bbox = [float(x) for x in lung_rel_bbox][1:]
    return lung_rel_bbox


def convert_yolo_to_abs(bbox, image):
    height, width = image.shape[:2]

    bbox_width = bbox[2] * width
    bbox_height = bbox[3] * height

    x_center = bbox[0] * width
    y_center = bbox[1] * height

    xmin = x_center - bbox_width / 2
    ymin = y_center - bbox_height / 2
    xmax = x_center + bbox_width / 2
    ymax = y_center + bbox_height / 2

    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    return np.array([xmin, ymin, xmax, ymax])


if __name__ == '__main__':
    main()

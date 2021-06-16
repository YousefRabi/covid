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
    parser.add_argument('--image_level_df')
    parser.add_argument('--opacity_mask_output_dir')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    original_images_dir = Path(args.original_images_dir)
    opacity_masks_dir = Path(args.opacity_mask_output_dir)
    opacity_masks_dir.mkdir(parents=True, exist_ok=True)

    image_df = pd.read_csv(args.image_level_df)

    for idx, row in tqdm(image_df.iterrows(), total=len(image_df)):
        image_id = row['id'].split('_')[0]

        original_image = skimage.io.imread(original_images_dir / (image_id + '.png'))

        if str(row['boxes']) == 'nan':
            opacity_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            skimage.io.imsave(opacity_masks_dir / (image_id + '.png'), opacity_mask)
            continue

        opacity_original_bboxes = read_opacity_label(row['boxes'])

        opacity_mask = make_mask(original_image, opacity_original_bboxes)

        skimage.io.imsave(opacity_masks_dir / (image_id + '.png'), opacity_mask)


def make_mask(image, opacity_bboxes_on_lungs):
    mask = np.zeros(image.shape, dtype=np.uint8)

    for opacity_bbox in opacity_bboxes_on_lungs:
        xmin, ymin, xmax, ymax = opacity_bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        mask[ymin:ymax, xmin:xmax] = 255

    return mask


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


if __name__ == '__main__':
    main()

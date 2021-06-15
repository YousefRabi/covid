from argparse import ArgumentParser
import tempfile
from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import cv2
import skimage.io

import numpy as np

from transforms import get_transforms
from datasets import get_dataloader
from utils.config import load_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = load_config(args.config)

    transforms = get_transforms(config.transforms.train)
    dataloader = get_dataloader(config, transforms, 'train')

    if Path('data/inspection_index.npy').exists():
        inspection_index = joblib.load('data/inspection_index.npy')
        print('Inspecting starting from ', inspection_index)
    else:
        inspection_index = 0

    for i, (image, mask, label, image_id) in enumerate(dataloader.dataset):
        if i < inspection_index:
            continue

        image = image.permute(1, 2, 0).numpy()
        mask = mask.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)
        ax.imshow(mask, alpha=0.2)
        with tempfile.TemporaryDirectory() as temp_dir:
            fig.savefig(Path(temp_dir) / 'temp_img.png')
            image = skimage.io.imread(Path(temp_dir) / 'temp_img.png')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('rendered image', image)
            key = cv2.waitKey(0)

            if key == ord("q"):
                joblib.dump(np.array([i]), 'data/inspection_index.npy')
                break


if __name__ == '__main__':
    main()

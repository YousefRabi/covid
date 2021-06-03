from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch

import numpy as np

from utils.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run_folders',
                        nargs='+',
                        help='Folders of different runs with different seeds to use in calculating noise.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    fold_scores = defaultdict(list)
    for run_folder in args.run_folders:
        for fold_folder in Path(run_folder).iterdir():
            best_model = torch.load(fold_folder / 'checkpoints' / 'best_model.pth')
            fold_name = fold_folder.stem.split('-')[1]
            fold_scores[fold_name].append(best_model['best_score'])

    log.info(f'fold_scores: {dict(fold_scores)}')

    fold_stds = [np.std(fold_scores[fold]) for fold in fold_scores]

    log.info(f'fold_stds: {fold_stds}')

    noise = np.mean(fold_stds)

    log.info(f'Noise: {noise:.5f}')


if __name__ == '__main__':
    main()

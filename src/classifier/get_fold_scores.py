from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--fold_models_dir')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    scores = {}
    for fold_model_path in Path(args.fold_models_dir).rglob('*.pth'):
        fold = fold_model_path.stem
        score = torch.load(fold_model_path)['best_score']
        scores[fold] = score

    for fold, score in scores.items():
        print(f'{fold} -- {score}')
    print(f'avg CV. -- {np.mean(list(scores.values()))}')


if __name__ == '__main__':
    main()

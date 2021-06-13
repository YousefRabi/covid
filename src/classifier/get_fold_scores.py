from argparse import ArgumentParser
from pathlib import Path
import shutil

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

    fold_models_dir = Path(args.fold_models_dir)
    all_folds_dir = fold_models_dir / 'all_folds'
    all_folds_dir.mkdir(parents=True, exist_ok=True)

    for fold_model_path in Path(args.fold_models_dir).rglob('*.pth'):
        if fold_model_path.stem == 'best_model':
            print('fold_model_path: ', fold_model_path)
            fold = fold_model_path.parent.parent.stem
            score = torch.load(fold_model_path)['best_score']
            shutil.copy(fold_model_path, all_folds_dir / f'{fold}.pth')
            scores[fold] = score

    for fold, score in scores.items():
        print(f'{fold} -- {score}')
    print(f'avg CV. -- {np.mean(list(scores.values()))}')


if __name__ == '__main__':
    main()

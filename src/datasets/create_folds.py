from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold


def create_kfold_split_uniform(input_path: Path, output_path: Path, folds: int = 5, seed: int = 42):
    if not output_path.exists():
        skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        train = pd.read_csv(input_path)

        train['fold'] = -1
        for i, (train_index, test_index) in enumerate(skf.split(train.index, train.StudyInstanceUID)):
            train.loc[test_index, 'fold'] = i
        train.to_csv(output_path, index=False)
        print('No folds: {}'.format(len(train[train['fold'] == -1])))

        for i in range(folds):
            part = train[train['fold'] == i]
            print(i, len(part))

    else:
        print('File already exists: {}'.format(output_path))


if __name__ == '__main__':
    input_path = Path('data/raw/train_image_level.csv')
    output_path = Path('data/processed/stratified_kfold_study_split_5_42.csv')

    create_kfold_split_uniform(input_path, output_path)

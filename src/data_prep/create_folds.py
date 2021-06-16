from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState

from sklearn.model_selection import StratifiedKFold


def create_stratified_group_kfold_split(image_df, study_df, folds: int = 5, seed: int = 42):
    image_study_df = create_image_study_df(image_df, study_df)
    print('image_df: ', len(image_df))
    cleaned_image_study_df = clean_image_study_df(image_df, image_study_df)
    print('cleaned_image_study_df: ', len(cleaned_image_study_df))

    cleaned_image_study_df['fold'] = -1
    cleaned_image_study_df['fold'] = stratified_group_k_fold(label='label',
                                                             group_column='study_id',
                                                             df=cleaned_image_study_df,
                                                             n_splits=folds,
                                                             seed=seed)

    print('No folds: {}'.format(len(cleaned_image_study_df[cleaned_image_study_df['fold'] == -1])))

    for fold in range(5):
        print(f'fold: {fold}')
        print(cleaned_image_study_df.label.value_counts())
        print('-' * 25)
        print(f'total: {len(cleaned_image_study_df[cleaned_image_study_df.fold == fold])}')
        print('*' * 50)

    return cleaned_image_study_df


def clean_image_study_df(image_df, image_study_df):
    image_level_none = image_df[image_df['label'] == 'none 1 0 0 1 1']
    study_label_dict = image_study_df[['study_id', 'label']].set_index('study_id').to_dict()['label']
    image_level_none['study_label'] = image_level_none['StudyInstanceUID'].map(study_label_dict)
    image_level_none = image_level_none.loc[
        (image_level_none['label'] == 'none 1 0 0 1 1') & (image_level_none['study_label'] != 0)]
    image_level_none['id'] = image_level_none['id'].apply(lambda x: x.replace('_image', ''))
    bad_image_ids = image_level_none['id'].values
    image_study_df = image_study_df[~image_study_df.image_id.isin(bad_image_ids)]
    return image_study_df


def create_image_study_df(image_df, study_df):
    image_study_df = pd.DataFrame(columns=['image_id', 'study_id', 'label'])

    image_study_df[['image_id', 'study_id']] = image_df[['id', 'StudyInstanceUID']]
    image_study_df['image_id'] = image_study_df['image_id'].apply(lambda x: x.replace('_image', ''))

    for idx, row in study_df.iterrows():
        label = row[['Negative for Pneumonia', 'Typical Appearance',
                     'Indeterminate Appearance', 'Atypical Appearance']].values
        label = np.argmax(label)
        image_study_df.loc[image_study_df['study_id'] == row['id'].replace('_study', ''), 'label'] = label

    return image_study_df


def stratified_group_k_fold(
        label: str,
        group_column: str,
        df: pd.DataFrame = None,
        file: str = None,
        n_splits=5,
        seed: int = 0
):
    random_state = RandomState(seed)

    if file is not None:
        df = pd.read_csv(file)

    labels = defaultdict(set)
    for g, l in zip(df[group_column], df[label]):
        labels[g].add(l)

    group_labels = dict()
    groups = []
    Y = []
    for k, v in labels.items():
        group_labels[k] = random_state.choice(list(v))
        Y.append(group_labels[k])
        groups.append(k)

    index = np.arange(len(group_labels))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=random_state).split(index, Y)

    group_folds = dict()
    for i, (train, val) in enumerate(folds):
        for j in val:
            group_folds[groups[j]] = i

    res = np.zeros(len(df))
    for i, g in enumerate(df[group_column]):
        res[i] = group_folds[g]

    return res.astype(np.int)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--image_csv')
    parser.add_argument('--study_csv')
    parser.add_argument('--output_csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if Path(args.output_csv).exists():
        raise ValueError(f'{args.output_csv} exists.')

    image_df = pd.read_csv(args.image_csv)
    study_df = pd.read_csv(args.study_csv)
    output_csv_path = Path(args.output_csv)

    stratified_image_study_df = create_stratified_group_kfold_split(image_df, study_df)

    stratified_image_study_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    main()

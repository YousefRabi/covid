from argparse import ArgumentParser

import pandas as pd
import numpy as np


def prepare_image_df(image_df, study_df):
    image_df = image_df[['id', 'StudyInstanceUID', 'fold']]
    image_df.columns = ['image_id', 'study_id', 'fold']
    image_df['image_id'] = image_df['image_id'].apply(lambda x: x.split('_')[0])

    study_df['id'] = study_df['id'].apply(lambda x: x.split('_')[0])
    study_df['label'] = -1

    study_df.loc[study_df['Negative for Pneumonia'] == 1, 'label'] = 0
    study_df.loc[study_df['Typical Appearance'] == 1, 'label'] = 1
    study_df.loc[study_df['Indeterminate Appearance'] == 1, 'label'] = 2
    study_df.loc[study_df['Atypical Appearance'] == 1, 'label'] = 3

    assert len(study_df.loc[study_df['label'] == -1]) == 0

    image_df['study_label'] = image_df['study_id']
    study_label_dict = dict(zip(study_df.id, study_df.label))
    image_df['study_label'] = image_df['study_label'].map(study_label_dict).astype(np.int32)

    assert image_df.study_label.nunique() == 4

    image_df.drop_duplicates(subset='study_id', keep=False, inplace=True, ignore_index=True)

    assert image_df.study_id.nunique() == len(image_df) == image_df.image_id.nunique()

    return image_df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--image_df_path')
    parser.add_argument('--study_df_path')
    parser.add_argument('--processed_image_df_path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    image_df = pd.read_csv(args.image_df_path)
    study_df = pd.read_csv(args.study_df_path)

    print('image_df len: ', len(image_df))
    print('study_df len: ', len(study_df))

    processed_image_df = prepare_image_df(image_df, study_df)

    print('processed_image_df len: ', len(processed_image_df))

    processed_image_df.to_csv(args.processed_image_df_path, index=False)


if __name__ == '__main__':
    main()

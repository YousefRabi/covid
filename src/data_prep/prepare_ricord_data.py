import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


def main():
    riccord_df = pd.read_csv('/home/yousef/deep-learning/kaggle/covid/data/external/MIDRC-RICORD-meta.csv')
    print('original riccord_df length: ', len(riccord_df))
    riccord_df = encode_labels(riccord_df)

    riccord_df['label'] = -1
    riccord_df.loc[riccord_df['Negative for Pneumonia'] == 1, 'label'] = 0
    riccord_df.loc[riccord_df['Typical Appearance'] == 1, 'label'] = 1
    riccord_df.loc[riccord_df['Indeterminate Appearance'] == 1, 'label'] = 2
    riccord_df.loc[riccord_df['Atypical Appearance'] == 1, 'label'] = 3

    assert len(riccord_df.loc[riccord_df['label'] == -1]) == 0

    riccord_df = riccord_df[['fname', 'label', 'study_id']]
    print('processed riccord_df length: ', len(riccord_df))

    print('riccord_df label value counts')
    print('*' * 50)
    print(riccord_df.label.value_counts())
    print('*' * 50)
    riccord_df.to_csv('/home/yousef/deep-learning/kaggle/covid/data/external/riccord_processed.csv', index=False)


def locate_row_to_delete(a):
    if a.max() <= 0.5:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    else:
        return a


def encode_labels(df):
    df = df[['fname', 'labels', 'StudyInstanceUID']]
    df.rename(columns={'StudyInstanceUID': 'study_id'}, inplace=True)
    df.dropna(subset=['labels'], inplace=True)

    # initialize label columns
    df['Negative for Pneumonia'] = 0
    df['Typical Appearance'] = 0
    df['Indeterminate Appearance'] = 0
    df['Atypical Appearance'] = 0

    # Count occurences of each category
    df['Negative for Pneumonia'] = df.labels.apply(lambda x: x.count('Negative'))
    df['Typical Appearance'] = df.labels.apply(lambda x: x.count('Typical'))
    df['Indeterminate Appearance'] = df.labels.apply(lambda x: x.count('Indeterminate'))
    df['Atypical Appearance'] = df.labels.apply(lambda x: x.count('Atypical'))

    # df to array for computations
    labels_np = df[
        ['Negative for Pneumonia', 'Typical Appearance', 'Atypical Appearance', 'Indeterminate Appearance']].values
    temp = labels_np / labels_np.sum(axis=1, keepdims=True)
    temp = np.apply_along_axis(locate_row_to_delete, 1, temp)

    temp -= 0.51

    temp[temp > 0] = 1
    temp[temp < 0] = 0

    df['Negative for Pneumonia'] = temp[:, 0]
    df['Typical Appearance'] = temp[:, 1]
    df['Indeterminate Appearance'] = temp[:, 2]
    df['Atypical Appearance'] = temp[:, 3]

    df.dropna(subset=['Negative for Pneumonia'], inplace=True)

    return df


if __name__ == '__main__':
    main()

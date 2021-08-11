from pathlib import Path
import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from map_boxes import mean_average_precision_for_boxes
from ensemble import get_labels_and_preds_arr


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--predictions_folder', help="Folder that contains experiment's oof")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    predictions_folder = Path(args.predictions_folder)

    filenames = os.listdir(predictions_folder)

    oof_filenames = np.sort([filename for filename in filenames if 'oof' in filename])
    oof_csvs = [pd.read_csv(predictions_folder / oof_filename) for oof_filename in oof_filenames]

    print(f'We have {len(oof_filenames)} oof files...')
    print()
    print(oof_filenames)

    oof_preds = np.zeros((len(oof_csvs[0]), 4, len(oof_filenames)))
    for i, oof_csv in enumerate(oof_csvs):
        oof_preds[:, :, i] = oof_csv[['negative', 'typical', 'indeterminate', 'atypical']].values

    oof_preds = np.mean(oof_preds, axis=-1)

    labels_arr, preds_arr = get_labels_and_preds_arr(oof_csvs[0].label.values, oof_preds, oof_csvs[0].study_id.values)

    mean_ap, _ = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=False)
    mean_ap = mean_ap * 2/3
    print(f'Average Ensemble has OOF mAP = {mean_ap:.4f}')


if __name__ == '__main__':
    main()

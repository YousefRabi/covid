from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from map_boxes import mean_average_precision_for_boxes

from classifier.utils.config import load_config
from classifier.predictors import Predictor
from classifier.utils.logconf import logging, formatter


log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--configs', nargs='+')
    parser.add_argument('--checkpoint_paths', nargs='+')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    folds_preds_dfs = []
    config_paths = sorted(list(args.configs))

    config = load_config(config_paths[0])
    file_handler = logging.FileHandler(Path(config.work_dir).parent / 'predict.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    checkpoint_paths = sorted(list(args.checkpoint_paths))
    log.info(f'config_paths: {config_paths}')
    log.info(f'checkpoint_paths: {checkpoint_paths}')

    oof = len(config_paths) > 1

    for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):
        config = load_config(config_path)
        predictor = Predictor(config, checkpoint_path)
        preds_df = predictor.run()

        if oof:
            folds_preds_dfs.append(preds_df)

    if oof:
        fold0, fold1, fold2, fold3, fold4 = folds_preds_dfs

        fold0 = fold0[fold0.fold == 0]
        fold1 = fold1[fold1.fold == 1]
        fold2 = fold2[fold2.fold == 2]
        fold3 = fold3[fold3.fold == 3]
        fold4 = fold4[fold4.fold == 4]

        folds_preds = pd.concat([fold0, fold1, fold2, fold3, fold4]).reset_index(drop=True)

        folds_preds.to_csv(args.output_path, index=False)

        labels_df = pd.read_csv(config.data.folds_df)
        labels_arr = get_labels_arr_from_df(labels_df)
        preds_arr = get_preds_arr_from_df(folds_preds)

        mean_ap, average_precisions = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=True)

        log.info(f'mean_ap: {mean_ap}')
        log.info(f'average_precisions: {average_precisions}')

    else:
        preds_df.to_csv(args.output_path, index=False)


def get_labels_arr_from_df(labels_df):
    labels_arr = []
    for study_id in labels_df.study_id.unique():
        label = labels_df.loc[labels_df.study_id == study_id, 'label'].unique().item()
        labels_arr.append([study_id, label, 0, 1, 0, 1])

    labels_arr = np.array(labels_arr)
    return labels_arr


def get_preds_arr_from_df(preds_df):
    preds_arr = []
    for study_id in preds_df.study_id.unique():
        pred_labels = preds_df.loc[preds_df.study_id == study_id][
            ['negative', 'typical', 'indeterminate', 'atypical']].values
        for i, pred_label in enumerate(pred_labels):
            for j, cls_pred in enumerate(pred_label):
                preds_arr.append([study_id, j, cls_pred, 0, 1, 0, 1])
    preds_arr = np.array(preds_arr)
    return preds_arr


if __name__ == '__main__':
    main()

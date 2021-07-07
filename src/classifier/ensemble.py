from pathlib import Path
import os
from tqdm import trange
from argparse import ArgumentParser

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from map_boxes import mean_average_precision_for_boxes


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
        oof_preds[:, :, i] = oof_csv.pred.values

    targets = oof_csvs[0].label.values
    study_ids = oof_csvs[0].study_id.values

    max_model_ix, max_model_score = get_max_model_ix_and_score(targets, oof_preds, study_ids)

    ensemble_model_indices, ensemble_model_weights, ensemble_score = get_best_ensemble(targets,
                                                                                       oof_preds,
                                                                                       max_model_ix,
                                                                                       max_model_score,)

    print(f'We are using models {ensemble_model_indices}')
    print(f'with weights {ensemble_model_weights}')
    print(f'and achieve ensemble mAP@0.5 = {ensemble_score:.4f}')

    ensemble_oof_preds = oof_preds[:, :, ensemble_model_indices[0]]
    for i, ensemble_model_ix in enumerate(ensemble_model_indices[1:]):
        ensemble_oof_preds = (ensemble_model_weights[i] * oof_preds[:, :, ensemble_model_ix] +
                              (1 - ensemble_model_weights[i]) * ensemble_oof_preds)


def get_max_model_ix_and_score(targets, oof_preds, study_ids):
    '''A function get the model ix with the highest OOF CV.

    :param targets: a NumPy array representing target values with shape (N, 4) where 4 is num_classes
    :param oof_preds: a NumPy array representing predictions of all models with shape (N, 4, num_models)
    :param study_ids: study id labels corresponding to rows in targets and oof_preds

    :return: The ix of the model with the highest OOF score and the highest score
    '''
    model_scores = []
    for i in range(oof_preds.shape[2]):
        preds_arr, labels_arr = get_preds_and_labels_arr(targets, oof_preds[:, :, i], study_ids)
        mean_ap = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=False)
        mean_ap = mean_ap * 2/3
        model_scores.append(mean_ap)
        print(f'Model {i} has OOF mAP = {mean_ap:.4f}')

    max_model_ix = np.argmax(model_scores)
    max_score = np.max(model_scores)
    return max_model_ix, max_score


def get_best_ensemble(targets,
                      oof_preds,
                      study_ids,
                      max_model_ix,
                      max_model_score,
                      improvement_thresh=0.0001,
                      weight_resolution=200,
                      patience=20,
                      duplicates=False):
    '''A function that searches for the best ensemble combination out of all the models whose
    predictions are represented third dim in oof_preds.

    It starts with an ensemble containing only the best model with the highest score, then
    loops over the remaining models twice. For every time a model is looped over,
    all other models are looped over and each one is attempted to be added to the ensemble
    using incremental weights. If resulting OOF CV increases by a minimum amount of improvement_thresh
    the model with its weights that caused it to increase are added to the ensemble. If not, next
    weight is tried until patience is reached, then model is skipped. The reason models
    are looped over twice is because each model might actually be beneficial after some other
    models are added.

    :param targets: a NumPy array representing target values with shape (N, 4)
    :param oof_preds: a NumPy array representing predictions of all models to be ensembled with shape (N, 4, num_models)
    :param study_ids: The study ids for each row of targets and oof_preds
    :param max_model_ix: Index of model with max OOF score.
    :param max_model_score: Maximum OOF score.
    :param improvement_thresh: The minimum increase in OOF score for model and weights to be added to ensemble.
    :param weight_resolution: The resolution of incremental weights to try and add model with.
    It starts with 1/weight_resolution and ends with weight_resolution/weight_resolution (1)
    :param patience: How much to wait to discard model if no increase in OOF score recorded.
    :param duplicates: Whether to add models more than once after having added them before.

    :return ensemble_model_indices, ensemble_model_weights, and max_ensemble_score
    '''

    print(f'Ensemble AUC = {max_model_score:.4f} by beginning with model {max_model_ix}')
    print()

    ensemble_models = [max_model_ix]
    model_weights = []

    for i in range(oof_preds.shape[2]):

        current_ensemble = oof_preds[:, :, max_model_ix]
        for j, model_ix in enumerate(ensemble_models[1:]):
            current_ensemble = model_weights[j] * oof_preds[:, :, model_ix] + (1 - model_weights[j]) * current_ensemble

        max_oof_score = 0
        add_model_ix = 0
        add_model_weights = 0
        print('Searching for best model to add...')

        progress_bar = trange(oof_preds.shape[2])
        for current_model_ix in progress_bar:
            progress_bar.set_description(f'current model ix: {current_model_ix}')
            if not duplicates and (current_model_ix in ensemble_models):
                continue

            current_weight = 0
            current_best_oof_score = 0
            count = 0
            for weight_res in range(weight_resolution):
                tmp = (weight_res / weight_resolution * oof_preds[:, :, current_model_ix] +
                       (1 - weight_res / weight_resolution) * current_ensemble)
                preds_arr, labels_arr = get_preds_and_labels_arr(targets, tmp, study_ids)
                mean_ap = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=False)
                mean_ap = mean_ap * 2/3

                if mean_ap > current_best_oof_score:
                    current_best_oof_score = mean_ap
                    current_weight = weight_res / weight_resolution

                else:
                    count += 1

                if count > patience:
                    break

            if current_best_oof_score > max_oof_score:
                max_oof_score = current_best_oof_score
                add_model_ix = current_model_ix
                add_model_weights = current_weight

        incremental_increase = max_oof_score - max_model_score
        if incremental_increase <= improvement_thresh:
            print()
            print('No increase. Stopping.')
            break

        print()
        print(f'Ensemble AUC = {max_oof_score:.4f} after adding model {add_model_ix} '
              f'with weight {add_model_weights:.3f}. '
              f'Increase of {incremental_increase:.4f}')

        max_model_score = max_oof_score
        ensemble_models.append(add_model_ix)
        model_weights.append(add_model_weights)

    return ensemble_models, model_weights, max_model_score


def get_preds_and_labels_arr(targets, oof_preds, study_ids):
    labels_arr = ([[study_id, value, 0, 1, 0, 1] for study_id, value in zip(study_ids, targets)])

    preds_arr = []
    for study_id, row in zip(study_ids, oof_preds):
        for i, cls_pred in enumerate(row):
            preds_arr.append([study_id, i, cls_pred, 0, 1, 0, 1])
    preds_arr = np.array(preds_arr)

    return labels_arr, preds_arr


if __name__ == '__main__':
    main()

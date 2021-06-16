import pandas as pd

import torch

from classifier.datasets.classification import StudyClassificationDataset


def test_init():
    image_df = pd.read_csv('data/processed/stratified_group_kfold_split_5_42.csv')
    dataset = StudyClassificationDataset('data/processed/train/lung_crops/lung',
                                         image_df,
                                         transforms=False,
                                         image_resolution=256)

    assert isinstance(dataset, StudyClassificationDataset)


def test_get_item():
    image_df = pd.read_csv('data/processed/stratified_group_kfold_split_5_42.csv')
    dataset = StudyClassificationDataset('data/processed/train/lung_crops/lung',
                                         image_df,
                                         transforms=False,
                                         image_resolution=256)

    item = dataset[0]

    assert isinstance(item[0], torch.Tensor)
    assert item[0].shape == (3, 256, 256)

    assert isinstance(item[1], torch.Tensor)
    assert item[1].dtype == torch.long
    assert 0 <= item[1].item() <= 3

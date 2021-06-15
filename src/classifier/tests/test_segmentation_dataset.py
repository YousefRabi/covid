import pandas as pd

import torch

from classifier.datasets.segmentation import StudySegmentationDataset


def test_init():
    image_df = pd.read_csv('data/processed/stratified_group_kfold_split_5_42.csv')
    dataset = StudySegmentationDataset('data/processed/train/lung_crops/lung',
                                       'data/processed/train/lung_crops/opacity_masks',
                                       image_df,
                                       transforms=False,
                                       image_resolution=256)

    assert isinstance(dataset, StudySegmentationDataset)


def test_get_item():
    image_df = pd.read_csv('data/processed/stratified_group_kfold_split_5_42.csv')
    dataset = StudySegmentationDataset('data/processed/train/lung_crops/lung',
                                       'data/processed/train/lung_crops/opacity_masks',
                                       image_df,
                                       transforms=False,
                                       image_resolution=256)

    item = dataset[0]

    image = item[0]
    mask = item[1]
    label = item[2]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)

    assert mask.max() <= 1
    assert mask.min() >= 0

    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert 0 <= label.item() <= 3

import pandas as pd

import torch

from classifier.datasets.classification import StudyClassificationDataset


def test_init():
    image_df = pd.read_csv('data/processed/processed_image_df.csv')
    dataset = StudyClassificationDataset('data/processed/train/png_div_2',
                                         image_df)

    assert isinstance(dataset, StudyClassificationDataset)


def test_get_item():
    image_df = pd.read_csv('data/processed/processed_image_df.csv')
    dataset = StudyClassificationDataset('data/processed/train/png_div_2',
                                         image_df)

    item = dataset[0]

    assert isinstance(item[0], torch.Tensor)
    assert item[0].shape == (3, 256, 256)

    assert isinstance(item[1], torch.Tensor)
    assert item[1].dtype == torch.long
    assert 0 <= item[1].item() <= 3

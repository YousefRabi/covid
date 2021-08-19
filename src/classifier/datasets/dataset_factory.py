import pandas as pd

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .classification import StudyClassificationDataset
from .segmentation import StudySegmentationDataset

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


def get_classification_dataset(config, split, transforms, folds_df):
    dataset = StudyClassificationDataset(config.data.data_dir,
                                         folds_df,
                                         transforms,
                                         config.data.image_resolution,
                                         config.train.overfit_single_batch)

    return dataset


def get_segmentation_dataset(config, split, transforms, folds_df):
    if config.data.external_data_df:
        external_data_df = pd.read_csv(config.data.external_data_df)
    else:
        external_data_df = False

    dataset = StudySegmentationDataset(config,
                                       folds_df,
                                       split,
                                       transforms,
                                       external_data_df)

    return dataset


def get_dataloader(config, transforms, split):
    DATASET_FUNC_DICT = {
        'StudyClassificationDataset': get_classification_dataset,
        'StudySegmentationDataset': get_segmentation_dataset
    }

    folds_df = pd.read_csv(config.data.folds_df)

    if split == 'train':
        folds_df = folds_df[folds_df.fold != config.data.idx_fold]
    elif split == 'valid':
        folds_df = folds_df[folds_df.fold == config.data.idx_fold]
    else:
        raise NotImplementedError(f'Split {split} is not implemented. Choose train or valid.')

    dataset = DATASET_FUNC_DICT[config.data.dataset_name](config,
                                                          split,
                                                          transforms,
                                                          folds_df)

    if split == 'train':
        sampler = RandomSampler(dataset)
        batch_size = config.train.batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.test.batch_size

    if config.rank == 0:
        log.info(f'Sampler: {sampler}')

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=config.num_workers,
                            drop_last=True,
                            pin_memory=True,)

    return dataloader

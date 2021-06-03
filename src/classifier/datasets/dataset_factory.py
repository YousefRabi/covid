import pandas as pd

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .classification import StudyClassificationDataset

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


def get_dataloader(config, transforms, split):
    folds_df = pd.read_csv(config.data.folds_df)

    if split == 'train':
        folds_df = folds_df[folds_df.fold != config.data.idx_fold]
    elif split == 'valid':
        folds_df = folds_df[folds_df.fold == config.data.idx_fold]
    else:
        raise NotImplementedError(f'Split {split} is not implemented. Choose train or valid.')

    dataset = globals().get(config.data.dataset_name)(config.data.data_dir,
                                                      folds_df,
                                                      transforms,
                                                      config.data.image_resolution,
                                                      config.train.overfit_single_batch)

    if split == 'train':
        sampler = RandomSampler(dataset)
        batch_size = config.train.batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = config.test.batch_size

    log.info(f'Sampler: {sampler}')

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=config.num_workers,
                            pin_memory=True,)

    return dataloader

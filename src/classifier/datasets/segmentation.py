from pathlib import Path

import skimage.io

import numpy as np
import pandas as pd

import torch

from classifier.utils.utils import img2tensor
from classifier.utils.logconf import logging
from data_prep.convert_dicom_to_png import resize_xray


log = logging.getLogger(__name__)


class StudySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_folder: str,
                 opacity_mask_folder: str,
                 image_df: pd.DataFrame,
                 split: str,
                 transforms,
                 image_resolution: int,
                 external_data_folder: str = '',
                 external_data_df: pd.DataFrame = False,
                 overfit_single_batch: bool = False):
        self.root = Path(image_folder)
        self.mask_folder = Path(opacity_mask_folder)
        self.image_resolution = image_resolution
        self.transforms = transforms

        self.image_paths = self.root.as_posix() + '/' + image_df.image_id.values + '.jpg'
        self.mask_paths = self.mask_folder.as_posix() + '/' + image_df.image_id.values + '.png'
        self.labels = image_df.label.values.tolist()
        self.study_ids = image_df.study_id.values.tolist()

        self.image_paths = self.image_paths.tolist()

        print('len(self.image_paths): ', len(self.image_paths))

        if external_data_folder and split == 'train':
            assert external_data_df is not False, 'You have to provide external df if specifying external_data_folder'
            external_image_fnames = external_data_df.fname.values
            external_image_paths = external_data_folder + '/' + external_image_fnames
            external_labels = external_data_df.label.values
            external_study_ids = external_data_df.study_id.values
            self.image_paths.extend(external_image_paths.tolist())
            self.labels.extend(external_labels.tolist())
            self.study_ids.extend(external_study_ids.tolist())

        print('len(self.image_paths): ', len(self.image_paths))

        if overfit_single_batch:
            self.image_paths = self.image_paths[:64]

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        study_id = self.study_ids[idx]

        try:
            image = skimage.io.imread(image_path)
            image = np.array(resize_xray(image, self.image_resolution))
        except Exception as e:
            log.error(f'Error {e} while reading image at {image_path}')
            raise

        try:
            mask_path = self.mask_paths[idx]
            mask = skimage.io.imread(mask_path)[..., 0]
            mask = np.array(resize_xray(mask, 32))
            mask_found = True
        except IndexError:
            mask_found = False
            mask = np.full((32, 32), 255)  # If no mask available for image

        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=2)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transforms:
            if mask_found:
                augment = self.transforms(image=image, mask=mask)
                image, mask = augment['image'], augment['mask']
            else:
                augment = self.transforms(image=image)
                image = augment['image']

        image = img2tensor(image) / 255
        mask = img2tensor(mask) / 255

        return image, mask, label, study_id

    def __len__(self):
        return len(self.image_paths)

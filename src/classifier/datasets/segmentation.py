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
                 transforms,
                 split,
                 image_resolution: int,
                 overfit_single_batch: bool = False):
        self.root = Path(image_folder)
        self.mask_folder = Path(opacity_mask_folder)
        self.image_resolution = image_resolution
        self.transforms = transforms

        self.image_df = image_df
        # self.image_df = self.image_df.loc[self.image_df.label.isin([1, 2, 3])]
        self.image_names = self.image_df.image_id.values + '.jpg'
        self.mask_names = self.image_df.image_id.values + '.png'
        self.labels = self.image_df.label.values
        self.study_ids = self.image_df.study_id.values

        self.log_study_ids = []

        for label in np.unique(self.labels):
            log_study_ids = np.random.choice(
                self.image_df.loc[self.image_df.label == label, 'study_id'].values, size=8 // 4, replace=False)
            while len(np.unique(log_study_ids)) == 1:
                log_study_ids = np.random.choice(
                    self.image_df.loc[self.image_df.label == label, 'study_id'].values, size=8 // 4, replace=False)
            self.log_study_ids.extend(log_study_ids)

        self.log_image_ids = self.image_df.loc[self.image_df.study_id.isin(self.log_study_ids), 'image_id'].values

        if overfit_single_batch:
            self.image_names = [image_id + '.jpg' for image_id in self.log_image_ids]
            self.mask_names = [image_id + '.png' for image_id in self.log_image_ids]
            self.labels = self.image_df.loc[self.image_df.image_id.isin(self.log_image_ids), 'label'].values
            self.study_ids = self.image_df.loc[self.image_df.image_id.isin(self.log_image_ids),
                                               'study_id'].values

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.root / self.image_names[idx]
        mask_path = self.mask_folder / self.mask_names[idx]
        study_id = self.study_ids[idx]

        try:
            image = skimage.io.imread(image_path)
            mask = skimage.io.imread(mask_path)
            image = np.array(resize_xray(image, self.image_resolution))
            mask = np.array(resize_xray(mask, self.image_resolution))
        except Exception as e:
            log.error(f'Error {e} while reading image at {image_path}')
            raise

        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=2)

        if mask.ndim == 3:
            mask = mask[..., 0]

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transforms:
            augment = self.transforms(image=image, mask=mask)

            if type(augment) == dict:  # Albumentations augment
                image = augment['image']
                mask = augment['mask']

        image = img2tensor(image) / 255
        mask = img2tensor(mask) / 255

        return image, mask, label, study_id

    def __len__(self):
        return len(self.image_names)

    def get_by_image_id(self, image_id):
        image_path = self.root / (image_id + '.jpg')
        mask_path = self.mask_folder / (image_id + '.png')
        study_id = self.image_df.loc[
            self.image_df.image_id == image_id, 'study_id'].item()
        label_id = self.image_df.loc[
            self.image_df.image_id == image_id, 'label'].item()

        try:
            image = skimage.io.imread(image_path)
            mask = skimage.io.imread(mask_path)
            image = np.array(resize_xray(image, self.image_resolution))
            mask = np.array(resize_xray(mask, self.image_resolution))
        except Exception as e:
            log.error(f'Error {e} while reading image at {image_path}')
            raise

        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=2)

        if mask.ndim == 3:
            mask = mask[..., 0]

        label = torch.tensor(label_id, dtype=torch.long)

        if self.transforms:
            augment = self.transforms(image=image, mask=mask)

            if type(augment) == dict:  # Albumentations augment
                image = augment['image']
                mask = augment['mask']

        image = img2tensor(image) / 255
        mask = img2tensor(mask) / 255

        return image, mask, label, study_id

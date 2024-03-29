from pathlib import Path

import skimage.io

import numpy as np
import pandas as pd

import torch

from classifier.utils.utils import img2tensor
from classifier.utils.logconf import logging
from data_prep.convert_dicom_to_png import resize_xray


log = logging.getLogger(__name__)


class StudyClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_folder: str,
                 image_df: pd.DataFrame,
                 transforms,
                 image_resolution: int,
                 overfit_single_batch: bool = False):
        self.root = Path(image_folder)
        self.image_resolution = image_resolution
        self.transforms = transforms

        self.image_names = image_df.image_id.values + '.jpg'
        self.labels = image_df.label.values
        self.study_ids = image_df.study_id.values

        if overfit_single_batch:
            self.image_names = self.image_names[:64]

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.root / self.image_names[idx]
        study_id = self.study_ids[idx]

        try:
            image = skimage.io.imread(image_path)
            image = np.array(resize_xray(image, self.image_resolution))
        except Exception as e:
            log.error(f'Error {e} while reading image at {image_path}')
            raise

        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=2)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transforms:
            image = self.transforms(image=image)

        if type(image) == dict:  # Albumentations augment
            image = image['image']

        image = img2tensor(image) / 255

        return image, label, study_id

    def __len__(self):
        return len(self.image_names)

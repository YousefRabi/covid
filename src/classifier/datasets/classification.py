from pathlib import Path
import glob

import skimage.io
import cv2

import numpy as np
import pandas as pd

import torch

from classifier.utils.utils import img2tensor

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


class StudyClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_folder: str,
                 image_df: pd.DataFrame,
                 transforms=False,
                 image_resolution: int = 256,
                 overfit_single_batch: bool = False):
        self.root = Path(image_folder)
        self.image_resolution = image_resolution
        self.transforms = transforms

        self.image_names = image_df.image_id.values + '.png'
        self.labels = image_df.label.values
        self.study_ids = image_df.study_id.values

        if overfit_single_batch:
            self.image_names = self.image_names[:64]

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.root / self.image_names[idx]
        study_id = self.study_ids[idx]

        try:
            image = skimage.io.imread(image_path)
            image = cv2.resize(image, (self.image_resolution, self.image_resolution))
        except Exception as e:
            log.error(f'Error {e} while reading image at {image_path}')
            raise

        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=2)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        image = img2tensor(image) / 255

        return image, label, study_id

    def __len__(self):
        return len(self.image_names)

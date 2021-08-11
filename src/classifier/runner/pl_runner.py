import os
from pathlib import Path
from easydict import EasyDict
from collections import defaultdict

import numpy as np

import skimage.io
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.core.lightning import LightningModule

from map_boxes import mean_average_precision_for_boxes

from classifier.models import get_model
from classifier.optimizers import get_optimizer
from classifier.datasets import get_dataloader
from classifier.transforms import get_transforms, get_first_place_melanoma_transforms, Mixup
from classifier.losses import LossBuilder
from classifier.schedulers import SchedulerBuilder
from classifier.utils.utils import (fix_seed, enumerate_with_estimate,
                                    save_model_with_optimizer, confusion_matrix_to_image)
from classifier.utils.logconf import logging, formatter


class LitModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = get_model(config)

        loss_builder = LossBuilder(config)
        self.cls_loss_func = loss_builder.get_loss()
        self.seg_loss_func = loss_builder.BCE()

    def forward(self, x, return_mask):
        return self.model(x, return_mask)

    def training_step(self, batch_tup, batch_idx):
        input_t, mask_t, label_t, study_id_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        mask_g = mask_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, mask_pred_g = self(input_g, return_mask=True)

        cls_loss_g = self.cls_loss_func(
            logits_g,
            label_g,
        )

        seg_loss_g = self.seg_loss_func(
            mask_pred_g,
            mask_g,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        return mean_loss


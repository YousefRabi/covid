import numpy as np

import torch
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import AveragePrecision


from classifier.models import get_model
from classifier.optimizers import get_optimizer
from classifier.datasets import get_dataloader
from classifier.transforms import get_transforms
from classifier.schedulers import SchedulerBuilder
from classifier.losses import LossBuilder


class LitModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = get_model(config)

        loss_builder = LossBuilder(config)
        self.cls_loss_func = loss_builder.get_loss()
        self.seg_loss_func = loss_builder.BCE()

        self.trn_transforms = get_transforms(config.transforms.train)
        self.val_transforms = get_transforms(config.transforms.test)

        self.average_precision = AveragePrecision(num_classes=4)

    def forward(self, x, return_mask=False):
        return self.model(x, return_mask)

    def training_step(self, batch_tup, batch_idx):
        training_step_outputs = {}

        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        self.log('loss/train', mean_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        training_step_outputs['loss'] = mean_loss
        training_step_outputs['probabilities'] = probabilities
        training_step_outputs['labels'] = labels

        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        probabilities = torch.cat(
            [training_step_output['probabilities'] for training_step_output in training_step_outputs])
        labels = torch.cat([training_step_output['labels'] for training_step_output in training_step_outputs])
        average_precisions = self.average_precision(probabilities, labels)
        negative, typical, indeterminate, atypical = average_precisions

        mean_average_precision = torch.mean(torch.stack(average_precisions))

        for cls, value in zip(['negative', 'typical', 'indeterminate', 'atypical'], average_precisions):
            self.log(f'map/train_{cls}',
                     value,
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     sync_dist=True)

        self.log('map/train',
                 mean_average_precision,
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 sync_dist=True)

    def validation_step(self, batch_tup, batch_idx):
        validation_step_outputs = {}

        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        self.log('loss/val', mean_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        validation_step_outputs['loss'] = mean_loss
        validation_step_outputs['probabilities'] = probabilities
        validation_step_outputs['labels'] = labels

        return validation_step_outputs

    def validation_epoch_end(self, validation_step_outputs):
        probabilities = torch.cat(
            [validation_step_output['probabilities'] for validation_step_output in validation_step_outputs])
        labels = torch.cat([validation_step_output['labels'] for validation_step_output in validation_step_outputs])
        average_precisions = self.average_precision(probabilities, labels)
        negative, typical, indeterminate, atypical = average_precisions

        mean_average_precision = torch.mean(torch.stack(average_precisions))

        for cls, value in zip(['negative', 'typical', 'indeterminate', 'atypical'], average_precisions):
            self.log(f'map/val_{cls}',
                     value,
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     sync_dist=True)

        self.log('map/val',
                 mean_average_precision,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False,
                 logger=True,
                 sync_dist=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        dataset_length = len(self.train_dataloader().dataset)
        scheduler_builder = SchedulerBuilder(optimizer, self.config, dataset_length)
        scheduler = scheduler_builder.get_scheduler()

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def train_dataloader(self):
        return get_dataloader(self.config, self.trn_transforms, 'train')

    def val_dataloader(self):
        return get_dataloader(self.config, self.val_transforms, 'valid')

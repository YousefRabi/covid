from collections import defaultdict

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
        self.automatic_optimization = False

        self.config = config
        self.model = get_model(config)

        loss_builder = LossBuilder(config)
        self.cls_loss_func = loss_builder.get_loss()
        self.seg_loss_func = loss_builder.BCE()

        self.trn_transforms = get_transforms(config.transforms.train)
        self.val_transforms = get_transforms(config.transforms.test)

        self.average_precision = AveragePrecision(num_classes=4)

        self.sync_dist = len(config.gpus) > 1

    def forward(self, x, return_mask=False):
        return self.model(x, return_mask)

    def training_step(self, batch_tup, batch_idx):
        training_step_outputs = {}

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        self.manual_backward(mean_loss)
        optimizer.step()
        scheduler.step()

        self.log(
            'loss/train', mean_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        training_step_outputs['study_ids'] = study_id_list
        training_step_outputs['loss'] = mean_loss
        training_step_outputs['probabilities'] = probabilities
        training_step_outputs['labels'] = labels

        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        study_id_preds = defaultdict(list)
        study_id_labels = dict()

        for training_step_output in training_step_outputs:
            for study_id, probabilities, label in zip(
                training_step_output['study_ids'],
                training_step_output['probabilities'],
                training_step_output['labels']
            ):
                study_id_preds[study_id].append(probabilities)
                study_id_labels[study_id] = label

        labels = []
        mean_preds = []

        for study_id, study_preds in study_id_preds.items():
            labels.append(study_id_labels[study_id])
            mean_preds.append(torch.from_numpy(np.mean(study_preds, axis=0)))

        mean_preds = torch.stack(mean_preds)
        labels = torch.tensor(labels)
        average_precisions = self.average_precision(mean_preds, labels)
        negative, typical, indeterminate, atypical = average_precisions

        mean_average_precision = torch.mean(torch.stack(average_precisions))

        for cls, value in zip(['negative', 'typical', 'indeterminate', 'atypical'], average_precisions):
            self.log(f'map/train_{cls}',
                     value,
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     sync_dist=self.sync_dist)

        self.log('map/train',
                 mean_average_precision,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=self.sync_dist)

    def validation_step(self, batch_tup, batch_idx):
        validation_step_outputs = {}

        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        self.log(
            'loss/val', mean_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        validation_step_outputs['study_ids'] = study_id_list
        validation_step_outputs['loss'] = mean_loss
        validation_step_outputs['probabilities'] = probabilities
        validation_step_outputs['labels'] = labels

        return validation_step_outputs

    def validation_epoch_end(self, validation_step_outputs):
        study_id_preds = defaultdict(list)
        study_id_labels = dict()

        for validation_step_output in validation_step_outputs:
            for study_id, probabilities, label in zip(
                validation_step_output['study_ids'],
                validation_step_output['probabilities'],
                validation_step_output['labels']
            ):
                study_id_preds[study_id].append(probabilities)
                study_id_labels[study_id] = label

        labels = []
        mean_preds = []

        for study_id, study_preds in study_id_preds.items():
            labels.append(study_id_labels[study_id])
            mean_preds.append(torch.from_numpy(np.mean(study_preds, axis=0)))

        mean_preds = torch.stack(mean_preds)
        labels = torch.tensor(labels)
        average_precisions = self.average_precision(mean_preds, labels)
        negative, typical, indeterminate, atypical = average_precisions

        mean_average_precision = torch.mean(torch.stack(average_precisions))

        for cls, value in zip(['negative', 'typical', 'indeterminate', 'atypical'], average_precisions):
            self.log(f'map/val_{cls}',
                     value,
                     on_step=False,
                     on_epoch=True,
                     logger=True,
                     sync_dist=self.sync_dist)

        self.log('map/val',
                 mean_average_precision,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=self.sync_dist)

        torch.save(self.model.state_dict(), 'runs/83/test_model_dict.pth')

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.config)
        dataset_length = len(self.train_dataloader().dataset)
        scheduler_builder = SchedulerBuilder(optimizer, self.config, dataset_length)
        scheduler = scheduler_builder.get_scheduler()

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
            }
        }

    def train_dataloader(self):
        return get_dataloader(self.config, self.trn_transforms, 'train')

    def val_dataloader(self):
        return get_dataloader(self.config, self.val_transforms, 'valid')

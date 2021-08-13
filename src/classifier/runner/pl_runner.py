import torch
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import AveragePrecision


from classifier.models import get_model
from classifier.optimizers import get_optimizer
from classifier.datasets import get_dataloader
from classifier.transforms import get_transforms, get_first_place_melanoma_transforms, Mixup
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
        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        average_precision = self.average_precision(probabilities, labels)

        self.log('train_loss', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            'train_ap', average_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return mean_loss

    def validation_step(self, batch_tup, batch_idx):
        inputs, masks, labels, study_id_list = batch_tup

        logits, mask_preds = self(inputs, return_mask=True)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()

        cls_loss_g = self.cls_loss_func(
            logits,
            labels,
        )

        seg_loss_g = self.seg_loss_func(
            mask_preds,
            masks,
        )

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        average_precision = self.average_precision(probabilities, labels)

        self.log('val_loss', mean_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(
            'val_ap', average_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return mean_loss

    def configure_optimizers(self):
        return get_optimizer(self.parameters(), self.config)

    def train_dataloader(self):
        return get_dataloader(self.config, self.trn_transforms, 'train')

    def val_dataloader(self):
        return get_dataloader(self.config, self.val_transforms, 'valid')

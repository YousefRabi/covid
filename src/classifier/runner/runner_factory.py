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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

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


log = logging.getLogger(__name__)

# Used for compute_batch_loss and log_metrics to index into metrics_t
METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_TN_NDX = 2
METRICS_FN_NDX = 3
METRICS_FP_NDX = 4
METRICS_SIZE = 5


id2label = {0: 'negative', 1: 'typical', 2: 'indeterminate', 3: 'atypical'}


class Runner:

    def __init__(self, config: EasyDict, rank=None):
        self.rank = rank
        self.config = config

        self.log_to_experiment_folder()

        fix_seed(self.config.seed)

        self.trn_writer = None
        self.val_writer = None
        self.init_tensorboard_writers()
        self.best_score = 0.0
        self.best_loss = float('inf')
        self.total_training_samples_count = 0

        self.ddp = (os.environ['CUDA_VISIBLE_DEVICES']) > 1
        if self.ddp:
            assert self.rank is not None, 'rank is None when DDP is True'

        self.device = torch.device('cuda')

        self.trn_transforms = get_transforms(config.transforms.train)
        log.info(f'train transforms: {self.trn_transforms}')
        self.val_transforms = get_transforms(config.transforms.test)
        self.mixup_fn = Mixup(**config.mixup) if config.mixup.prob > 0 else None

        self.train_dl = self.init_train_dl()
        self.val_dl = self.init_val_dl()

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        loss_builder = LossBuilder(config)
        self.cls_loss_func = loss_builder.get_loss()
        self.seg_loss_func = loss_builder.BCE()

        scheduler_builder = SchedulerBuilder(self.optimizer,
                                             config,
                                             len(self.train_dl.dataset))
        self.scheduler = scheduler_builder.get_scheduler()

        self.scaler = torch.cuda.amp.GradScaler()

        if self.config.train.checkpoint_path:
            self.load_checkpoint()

    def log_to_experiment_folder(self):
        self.config.work_dir = Path(self.config.work_dir)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.config.work_dir / 'train.log')

        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        log.addHandler(file_handler)

    def init_model(self):
        model = get_model(self.config)
        log.info("Using CUDA; current_device: {}.".format(torch.cuda.current_device()))
        if self.ddp:
            model = model.to(self.rank)
            model = DDP(model, device_ids=[self.rank])
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = model.to(self.device)

        return model

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.config.train.checkpoint_path,
                                    map_location=self.config.device)

            if self.config.multi_gpu:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.optimizer.params.lr

            self.best_score = checkpoint['best_score']

            print('Loaded model from checkpoint: ', self.config.train.checkpoint_path)
            print('Best score: ', self.best_score)
            print('*' * 50)

        except Exception as e:
            print('Exception loading checkpoint: ', e)

    def init_optimizer(self):
        return get_optimizer(self.model.parameters(), self.config)

    def init_train_dl(self):
        return get_dataloader(self.config, self.trn_transforms, 'train')

    def init_val_dl(self):
        return get_dataloader(self.config, self.val_transforms, 'valid')

    def init_tensorboard_writers(self):
        if self.trn_writer is None:
            log_dir = os.path.join(
                'runs',
                self.config.experiment_version,
                f'fold-{self.config.data.idx_fold}',
                self.config.exp_name
            )

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn')
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val')

    def run(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.config))

        score = 0.0

        if self.config.scheduler.warmup.apply:
            self.optimizer.zero_grad()
            self.optimizer.step()

        for epoch_ndx in range(1, self.config.train.num_epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}".format(
                epoch_ndx,
                self.config.train.num_epochs,
                len(self.train_dl),
                len(self.val_dl),
                self.config.train.batch_size,
            ))

            trn_metrics_t, labels_arr, preds_arr, confusion_matrix_dict = self.do_training(epoch_ndx)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t, labels_arr, preds_arr, confusion_matrix_dict)

            if not self.config.train.overfit_single_batch:
                val_metrics_t, labels_arr, preds_arr, confusion_matrix_dict = self.do_validation(epoch_ndx)
                score = self.log_metrics(epoch_ndx, 'val', val_metrics_t, labels_arr, preds_arr, confusion_matrix_dict)

                if score > self.best_score:
                    log.info(f'mAP improved from {self.best_score:.6f} -> {score:.6f}. Saving model.')
                    save_model_with_optimizer(self.model,
                                              self.optimizer,
                                              self.scheduler,
                                              score,
                                              self.config.multi_gpu,
                                              self.config.work_dir / 'checkpoints' / 'best_model.pth')
                    self.best_score = score

        save_model_with_optimizer(self.model,
                                  self.optimizer,
                                  self.scheduler,
                                  score,
                                  self.config.multi_gpu,
                                  self.config.work_dir / 'checkpoints' / 'latest_model.pth')

        log.info(f'Best mAP: {self.best_score}')
        self.trn_writer.close()
        self.val_writer.close()

        trn_loss = trn_metrics_t[METRICS_LOSS_NDX].mean()

        if not self.config.train.overfit_single_batch:
            val_loss = val_metrics_t[METRICS_LOSS_NDX].mean()
            return trn_loss, val_loss

        return trn_loss

    def do_training(self, epoch_ndx):
        if self.config.scheduler.warmup.apply:
            self.scheduler.step(epoch_ndx)

        self.model.train()
        trn_metrics_t = torch.zeros(
            METRICS_SIZE,
            len(self.train_dl.dataset),
            device=torch.device('cpu'),
        )

        labels_dict = defaultdict()
        preds_dict = defaultdict(list)
        confusion_matrix_dict = {'labels': [], 'preds': []}

        log.info(f'LR - {self.optimizer.param_groups[0]["lr"]}')
        batch_iter = enumerate_with_estimate(
            self.train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=self.train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                epoch_ndx,
                batch_ndx,
                batch_tup,
                self.train_dl.batch_size,
                trn_metrics_t,
                labels_dict,
                preds_dict,
                confusion_matrix_dict,
                'trn'
            )

            self.scaler.scale(loss_var).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.scheduler.name in ['onecycle', 'cosine']:
                self.scheduler.step()

        self.total_training_samples_count += len(self.train_dl.dataset)

        return trn_metrics_t.to('cpu'), labels_dict, preds_dict, confusion_matrix_dict

    def do_validation(self, epoch_ndx):
        with torch.no_grad():
            self.model.eval()
            val_metrics_t = torch.zeros(
                METRICS_SIZE,
                len(self.val_dl.dataset),
                device=torch.device('cpu'),
            )

            labels_dict = dict()
            preds_dict = defaultdict(list)
            confusion_matrix_dict = {'labels': [], 'preds': []}

            batch_iter = enumerate_with_estimate(
                self.val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=self.val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    epoch_ndx,
                    batch_ndx,
                    batch_tup,
                    self.val_dl.batch_size,
                    val_metrics_t,
                    labels_dict,
                    preds_dict,
                    confusion_matrix_dict,
                    'val',
                )

        return val_metrics_t, labels_dict, preds_dict, confusion_matrix_dict

    def compute_batch_loss(self, epoch_ndx, batch_ndx, batch_tup, batch_size,
                           metrics_t, labels_dict, preds_dict, confusion_matrix_dict, mode_str):
        input_t, mask_t, label_t, study_id_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        mask_g = mask_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.mixup_fn is not None and mode_str == 'trn':
            input_g, mask_g, label_g = self.mixup_fn(input_g, mask_g, label_g)

        study_id_arr = np.array(study_id_list)

        with autocast():
            logits_g, mask_pred_g = self.model(input_g, return_mask=True)
            probability_arr = torch.nn.functional.softmax(logits_g, dim=-1).cpu().detach().numpy()
            preds = torch.argmax(logits_g, dim=-1).cpu().detach().numpy().ravel().tolist()
            confusion_matrix_dict['preds'].extend(preds)
            confusion_matrix_dict['labels'].extend(label_g.cpu().detach().numpy().ravel().tolist())

            cls_loss_g = self.cls_loss_func(
                logits_g,
                label_g,
            )

            seg_loss_g = self.seg_loss_func(
                mask_pred_g,
                mask_g,
            )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + batch_size

        for i, study_id in enumerate(study_id_arr):
            preds_dict[study_id].append(probability_arr[i])
            labels_dict[study_id] = label_g.cpu().detach().numpy()[i]
            if study_id in self.train_dl.dataset.log_study_ids:
                self.log_image(epoch_ndx, input_g[i], mask_g[i], label_g[i], study_id,
                               mask_pred_g[i], logits_g[i], mode_str)

        metrics_t[METRICS_LOSS_NDX, start_ndx:end_ndx] = cls_loss_g

        mean_loss = cls_loss_g.mean() + self.config.loss.params.seg_multiplier * seg_loss_g.mean()

        return mean_loss

    def log_image(self, epoch_ndx, image, mask, label, study_id,
                  attention_map, logits, mode_str):
        prediction = torch.argmax(logits)

        writer = getattr(self, mode_str + '_writer')

        if self.mixup_fn is not None:
            label = str(label.detach().cpu().numpy().tolist())
        else:
            label = id2label[label.detach().cpu().item()]

        prediction = id2label[prediction.item()]

        if epoch_ndx == 1:
            heatmap_image_label = self._combine_heatmap_with_image(
                image, mask, label)

            writer.add_image(
                f'{mode_str}/{study_id}_label',
                heatmap_image_label,
                self.total_training_samples_count,
                dataformats='HWC',
            )

        heatmap_image_pred = self._combine_heatmap_with_image(
            image, attention_map, prediction)

        writer.add_image(
            f'{mode_str}/{study_id}_pred',
            heatmap_image_pred,
            self.total_training_samples_count,
            dataformats='HWC',
        )

        writer.flush()

    def _combine_heatmap_with_image(self,
                                    image,
                                    heatmap,
                                    label_name,
                                    font_scale=1.5,
                                    font_name=cv2.FONT_HERSHEY_SIMPLEX,
                                    font_color=(255, 255, 255),
                                    font_pixel_width=1):
        # get the min and max values once to be used with scaling
        min_val = heatmap.min()
        max_val = heatmap.max()

        # Scale the heatmap in range 0-255
        heatmap = 255 - (255 * (heatmap - min_val)) / (max_val - min_val + 1e-10)
        heatmap = heatmap.data.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Scale the image as well
        scaled_image = image * 255.0
        scaled_image = scaled_image.cpu().numpy().astype(np.uint8).transpose((1, 2, 0))

        if scaled_image.shape[2] == 1:
            scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2RGB)

        # generate the heatmap
        heatmap_image = cv2.addWeighted(scaled_image, 0.7, heatmap, 0.3, 0)

        # superimpose label_name
        (_, text_size_h), baseline = cv2.getTextSize(label_name, font_name, font_scale, font_pixel_width)
        heatmap_image = cv2.putText(heatmap_image, label_name,
                                    (10, text_size_h + baseline + 20),
                                    font_name,
                                    font_scale,
                                    font_color,
                                    thickness=font_pixel_width)
        return heatmap_image

    def log_metrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        labels_dict,
        preds_dict,
        confusion_matrix_dict,
    ):
        # assert torch.isfinite(metrics_t).all()

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        if mode_str == 'val':
            labels_arr = np.array([[key, value, 0, 1, 0, 1] for key, value in labels_dict.items()])

            preds_arr = []
            for key, value in preds_dict.items():
                mean_preds = np.mean(value, axis=0)
                for i, cls_pred in enumerate(mean_preds):
                    preds_arr.append([key, i, cls_pred, 0, 1, 0, 1])
            preds_arr = np.array(preds_arr)

            mean_ap, average_precisions = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=False)
            # Multiply by 2 /3 because LB metric is mAP and study ids have 4 labels and image ids have 2 labels
            metrics_dict['map/negative'] = average_precisions['0'][0]
            metrics_dict['map/typical'] = average_precisions['1'][0]
            metrics_dict['map/indeterminate'] = average_precisions['2'][0]
            metrics_dict['map/atypical'] = average_precisions['3'][0]
            metrics_dict['map/all'] = mean_ap

            log.info(
                ("E{} {:8} {map/all:.4f} mAP@0.5, "
                 + "{map/negative:.4f} mAP/negative, "
                 + "{map/typical:.4f} mAP/typical "
                 + "{map/indeterminate:.4f} mAP/indeterminate "
                 + "{map/atypical:.4f} mAP/atypical "
                 ).format(
                    epoch_ndx,
                    mode_str,
                    **metrics_dict,
                )
            )

        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'cls_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.total_training_samples_count)

        if mode_str == 'val':

            cm = confusion_matrix(confusion_matrix_dict['labels'], confusion_matrix_dict['preds'])
            confusion_matrix_image = confusion_matrix_to_image(cm)

            writer.add_image(
                f'{mode_str}-confusion-matrix',
                confusion_matrix_image,
                self.total_training_samples_count,
                dataformats='HWC',
            )

        writer.flush()

        if mode_str == 'trn':
            return metrics_dict['loss/all']

        return metrics_dict['map/all']

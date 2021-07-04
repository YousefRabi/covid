from collections import defaultdict
import os
import gc
from pathlib import Path
from easydict import EasyDict

import numpy as np

import skimage.io
from sklearn.metrics import confusion_matrix
import cv2

import torch
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from map_boxes import mean_average_precision_for_boxes

from data_prep.convert_dicom_to_png import resize_xray
from classifier.models import get_model
from classifier.models.gain import LOGITS, ATTENTION_MAPS, I_STAR, LOGITS_AM
from classifier.optimizers import get_optimizer
from classifier.datasets import get_dataloader
from classifier.transforms import get_transforms
from classifier.losses import LossBuilder
from classifier.schedulers import SchedulerBuilder
from classifier.utils.utils import fix_seed, enumerate_with_estimate, save_model_with_optimizer, confusion_matrix_to_image
from classifier.utils.logconf import logging, formatter


log = logging.getLogger(__name__)

# Used for compute_batch_loss and log_metrics to index into metrics_t
METRICS_CLS_LOSS_NDX = 0
METRICS_ATTN_MINING_LOSS_NDX = 1
METRICS_EXT_SUPERVISION_LOSS_NDX = 2
METRICS_TOTAL_LOSS_NDX = 3
METRICS_TP_NDX = 4
METRICS_TN_NDX = 5
METRICS_FN_NDX = 6
METRICS_FP_NDX = 7
METRICS_SIZE = 8

id2label = {0: 'negative', 1: 'typical', 2: 'indeterminate', 3: 'atypical'}


class Runner:

    def __init__(self, config: EasyDict):
        self.config = config

        self.log_to_experiment_folder()

        fix_seed(self.config.seed)

        self.trn_writer = None
        self.val_writer = None
        self.init_tensorboard_writers()
        self.best_score = 0.0
        self.best_loss = float('inf')
        self.total_training_samples_count = 0

        self.device = torch.device('cuda')

        self.trn_transforms = get_transforms(config.transforms.train)
        self.val_transforms = get_transforms(config.transforms.test)

        self.train_dl = self.init_train_dl()
        self.val_dl = self.init_val_dl()

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        loss_builder = LossBuilder(config)
        self.cls_loss_func = loss_builder.get_loss()
        self.attn_mining_loss = loss_builder.attention_mining_loss()
        self.extra_supervision_loss = loss_builder.extra_supervision_loss()

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
        if self.config.multi_gpu:
            model = torch.nn.DataParallel(model)
        model = model.to(self.device)

        return model

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.config.train.checkpoint_path,
                                    map_location=self.config.device)
        except Exception as e:
            log.info(f'Exception {e} when loading checkpoint.')
            raise

        try:
            if self.config.multi_gpu:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

        except Exception as e:
            log.info(f'Exception {e} when loading model state dict.')
            raise

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.optimizer.params.lr
        except Exception as e:
            log.info(f'Exception {e} when loading optimizer state dict.')

        try:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            log.info(f'Exception {e} when loading scheduler state dict.')

        try:
            self.best_score = checkpoint['best_score']
        except KeyError:
            log.info('No best score recorded in checkpoint.')

        print('Loaded model from checkpoint: ', self.config.train.checkpoint_path)
        print('Best score: ', self.best_score)
        print('*' * 50)

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
        for epoch_ndx in range(1, self.config.train.num_epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}".format(
                epoch_ndx,
                self.config.train.num_epochs,
                len(self.train_dl),
                len(self.val_dl),
                self.config.train.batch_size,
            ))

            log.info(f'LR - {self.optimizer.param_groups[0]["lr"]}')
            trn_metrics_t, labels_arr, preds_arr, confusion_matrix_dict = self.do_training(epoch_ndx)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t, labels_arr, preds_arr, confusion_matrix_dict)

            val_metrics_t, labels_arr, preds_arr, confusion_matrix_dict = self.do_validation(epoch_ndx)
            score = self.log_metrics(epoch_ndx, 'val', val_metrics_t, labels_arr, preds_arr, confusion_matrix_dict)

            if score > self.best_score:
                log.info(f'Score improved from {self.best_score:.6f} -> {score:.6f}. Saving model.')
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

        log.info(f'Best score: {self.best_score}')
        self.trn_writer.close()
        self.val_writer.close()

        trn_loss = trn_metrics_t[METRICS_TOTAL_LOSS_NDX].mean()

        if not self.config.train.overfit_single_batch:
            val_loss = val_metrics_t[METRICS_TOTAL_LOSS_NDX].mean()
            return trn_loss, val_loss

        return trn_loss

    def do_training(self, epoch_ndx):
        self.model.train()
        trn_metrics_t = torch.zeros(
            METRICS_SIZE,
            len(self.train_dl.dataset),
            device=torch.device('cpu')
        )

        labels_dict = defaultdict()
        preds_dict = defaultdict(list)
        confusion_matrix_dict = {'labels': [], 'preds': []}

        batch_iter = enumerate_with_estimate(
            self.train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=self.train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            self.do_one_step(
                epoch_ndx,
                batch_ndx,
                batch_tup,
                self.train_dl.batch_size,
                trn_metrics_t,
                labels_dict,
                preds_dict,
                'trn',
                confusion_matrix_dict
            )

            if self.config.scheduler.name == 'onecycle':
                self.scheduler.step()

        self.total_training_samples_count += len(self.train_dl.dataset)

        return trn_metrics_t, labels_dict, preds_dict, confusion_matrix_dict

    def do_validation(self, epoch_ndx):
        with torch.no_grad():
            self.model.eval()
            val_metrics_t = torch.zeros(
                METRICS_SIZE,
                len(self.val_dl.dataset),
                device=torch.device('cpu')
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
                self.do_one_step(
                    epoch_ndx,
                    batch_ndx,
                    batch_tup,
                    self.val_dl.batch_size,
                    val_metrics_t,
                    labels_dict,
                    preds_dict,
                    'val',
                    confusion_matrix_dict

                )

        return val_metrics_t, labels_dict, preds_dict, confusion_matrix_dict

    def do_one_step(self, epoch_ndx, batch_ndx, batch_tup, batch_size,
                    metrics_t, labels_dict, preds_dict, mode_str, confusion_matrix_dict):
        input_t, mask_t, label_t, study_id_list = batch_tup

        input_g = input_t.to(self.device)
        mask_g = mask_t.to(self.device)
        label_g = label_t.to(self.device)

        study_id_arr = np.array(study_id_list)

        with autocast():
            results = self.model(input_g, label_g, mode_str)
            probability_arr = torch.nn.functional.softmax(results[LOGITS], dim=-1).cpu().detach().numpy()
            preds = torch.argmax(results['logits'], dim=-1).cpu().detach().numpy().ravel().tolist()
            confusion_matrix_dict['preds'].extend(preds)
            confusion_matrix_dict['labels'].extend(label_g.cpu().detach().numpy().ravel().tolist())

        cls_loss_g = self.cls_loss_func(
            results[LOGITS],
            label_g,
        )

        if mode_str == 'trn':
            # label_one_hot_g = torch.nn.functional.one_hot(label_g, num_classes=4)

            # attention_mining_loss_g = self.attn_mining_loss(
            #     results[LOGITS_AM],
            #     label_one_hot_g,
            # )

            masks = []
            attention_maps = []
            for i, mask in enumerate(mask_t):
                if mask.sum() != 0:
                    masks.append(mask)
                    attention_maps.append(results[ATTENTION_MAPS][i])

            masks = torch.stack(masks).to(self.device)
            attention_maps = torch.stack(attention_maps).to(self.device)

            ext_loss_g = self.extra_supervision_loss(
                attention_maps,
                masks,
            )

            ext_loss_g = ext_loss_g.mean()

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + batch_size

        for i, study_id in enumerate(study_id_arr):
            preds_dict[study_id].append(probability_arr[i])
            labels_dict[study_id] = label_g.cpu().detach().numpy()[i]
            if study_id in self.train_dl.dataset.log_study_ids:
                self.log_image(epoch_ndx, input_g[i], mask_g[i], label_g[i], study_id,
                               results['attention_maps'][i], results['logits'][i], mode_str)

        metrics_t[METRICS_CLS_LOSS_NDX, start_ndx:end_ndx] = cls_loss_g.detach().cpu()

        total_loss = cls_loss_g.mean()

        if mode_str == 'trn':
            total_loss += self.config.model.params.omega * ext_loss_g   # +\
            # self.config.model.params.alpha * attention_mining_loss_g

        if mode_str == 'trn':
            # metrics_t[METRICS_ATTN_MINING_LOSS_NDX, start_ndx:end_ndx] = attention_mining_loss_g.detach().cpu()
            metrics_t[METRICS_EXT_SUPERVISION_LOSS_NDX, start_ndx:end_ndx] = ext_loss_g.detach().cpu()

        metrics_t[METRICS_TOTAL_LOSS_NDX, start_ndx:end_ndx] = total_loss.cpu()

        if mode_str == 'trn':
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def log_image(self, epoch_ndx, image, mask, label, study_id,
                  attention_map, logits, mode_str):
        prediction = torch.argmax(logits)

        writer = getattr(self, mode_str + '_writer')

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

        if mode_str == 'trn':
            metrics_dict['loss/cls'] = metrics_t[METRICS_CLS_LOSS_NDX].mean()
            # metrics_dict['loss/attn'] = metrics_t[METRICS_ATTN_MINING_LOSS_NDX].mean()
            metrics_dict['loss/ext'] = metrics_t[METRICS_EXT_SUPERVISION_LOSS_NDX].mean()

        metrics_dict['loss/all'] = metrics_t[METRICS_TOTAL_LOSS_NDX].mean()

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
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{map/all:.4f} mAP@0.5 "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        if mode_str == 'trn':
            log.info(
                ("E{} {:8} {loss/cls:.4f} cls_loss, "
                 # + "{loss/attn:.4f} attn_loss "
                 + "{loss/ext:.4f} ext_loss"
                 ).format(
                    epoch_ndx,
                    mode_str,
                    **metrics_dict,
                )
            )

        log.info(
            ("E{} {:8} {map/negative:.4f} mAP/negative, "
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

        cm = confusion_matrix(confusion_matrix_dict['labels'], confusion_matrix_dict['preds'])
        confusion_matrix_image = confusion_matrix_to_image(cm)

        writer.add_image(
            f'{mode_str}-confusion-matrix',
            confusion_matrix_image,
            self.total_training_samples_count,
            dataformats='HWC',
        )

        if metrics_dict['map/all'] > self.best_score:
            writer.add_image(
                f'{mode_str}-best-confusion-matrix',
                confusion_matrix_image,
                self.total_training_samples_count,
                dataformats='HWC',
            )

        writer.flush()

        return metrics_dict['map/all']

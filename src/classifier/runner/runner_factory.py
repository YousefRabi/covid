import os
from pathlib import Path
from easydict import EasyDict

import numpy as np

import skimage.io

import torch
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from map_boxes import mean_average_precision_for_boxes

from classifier.models import get_model
from classifier.optimizers import get_optimizer
from classifier.datasets import get_dataloader
from classifier.transforms import get_transforms
from classifier.losses import LossBuilder
from classifier.schedulers import SchedulerBuilder
from classifier.utils.utils import fix_seed, enumerate_with_estimate, save_model_with_optimizer
from classifier.utils.logconf import logging, formatter


log = logging.getLogger(__name__)

# Used for compute_batch_loss and log_metrics to index into metrics_t
METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_TN_NDX = 2
METRICS_FN_NDX = 3
METRICS_FP_NDX = 4
METRICS_SIZE = 5


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

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        loss_builder = LossBuilder(config)
        self.loss_func = loss_builder.get_loss()

        scheduler_builder = SchedulerBuilder(self.optimizer, config)
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

            if self.config.multi_gpu:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.optimizer.params.lr

            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

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

        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        score = 0.0
        for epoch_ndx in range(1, self.config.train.num_epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}".format(
                epoch_ndx,
                self.config.train.num_epochs,
                len(train_dl),
                len(val_dl),
                self.config.train.batch_size,
            ))

            trn_metrics_t, labels_arr, preds_arr = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', trn_metrics_t, labels_arr, preds_arr)

            if not self.config.train.overfit_single_batch:
                val_metrics_t, labels_arr, preds_arr = self.do_validation(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', val_metrics_t, labels_arr, preds_arr)

                if score > self.best_score:
                    log.info(f'Loss improved from {self.best_score:.6f} -> {score:.6f}. Saving model.')
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

        trn_loss = trn_metrics_t[METRICS_LOSS_NDX].mean()

        if not self.config.train.overfit_single_batch:
            val_loss = val_metrics_t[METRICS_LOSS_NDX].mean()
            return trn_loss, val_loss

        return trn_loss

    def do_training(self, epoch_ndx, train_dl):
        self.model.train()
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )
        labels_list = []
        preds_list = []

        batch_iter = enumerate_with_estimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trn_metrics_g,
                labels_list,
                preds_list
            )

            self.scaler.scale(loss_var).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.total_training_samples_count += len(train_dl.dataset)

        labels_arr = np.array(labels_list, dtype='object')
        preds_arr = np.array(preds_list, dtype='object')

        return trn_metrics_g.to('cpu'), labels_arr, preds_arr

    def do_validation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            labels_list = []
            preds_list = []

            batch_iter = enumerate_with_estimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics_g,
                    labels_list,
                    preds_list
                )

            labels_arr = np.array(labels_list)
            preds_arr = np.array(preds_list)

        return val_metrics_g.to('cpu'), labels_arr, preds_arr

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g, labels_list, preds_list):
        input_t, label_t, study_id_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        study_id_arr = np.array(study_id_list)
        labels_arr = np.empty((label_g.shape[0], 6), dtype='object')
        preds_arr = np.empty((input_g.shape[0] * 4, 7), dtype='object')

        with autocast():
            logits_g = self.model(input_g)
            probability_arr = torch.nn.functional.softmax(logits_g, dim=-1).cpu().detach().numpy()

            loss_g = self.loss_func(
                logits_g,
                label_g,
            )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + batch_size

        with torch.no_grad():
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

            labels_arr[:, 0] = study_id_arr
            labels_arr[:, 1] = label_g.cpu().detach().numpy()
            labels_arr[:, 2:] = [0, 1, 0, 1]

            preds_arr[:, 0] = np.concatenate((study_id_arr,) * 4)
            preds_arr[:, 3:] = [0, 1, 0, 1]
            preds_arr[:len(study_id_arr), 1] = 0
            preds_arr[:len(study_id_arr), 2] = probability_arr[:, 0]
            preds_arr[len(study_id_arr):2*len(study_id_arr), 1] = 1
            preds_arr[len(study_id_arr):2*len(study_id_arr), 2] = probability_arr[:, 1]
            preds_arr[2*len(study_id_arr):3*len(study_id_arr), 1] = 2
            preds_arr[2*len(study_id_arr):3*len(study_id_arr), 2] = probability_arr[:, 2]
            preds_arr[3*len(study_id_arr):4*len(study_id_arr), 1] = 3
            preds_arr[3*len(study_id_arr):4*len(study_id_arr), 2] = probability_arr[:, 3]

        labels_list.extend(labels_arr.tolist())
        preds_list.extend(preds_arr.tolist())

        return loss_g.mean()

    def log_metrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        labels_arr,
        preds_arr,
    ):
        # assert torch.isfinite(metrics_t).all()

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()

        mean_ap, average_precisions = mean_average_precision_for_boxes(labels_arr, preds_arr, verbose=False)
        # Multiply by 2 /3 because LB metric is mAP and study ids have 4 labels and image ids have 2 labels
        metrics_dict['map/all'] = mean_ap * 2 / 3

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{map/all:.4f} mAP@0.5"
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

        writer.flush()

        return metrics_dict['map/all']

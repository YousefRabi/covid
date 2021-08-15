from easydict import EasyDict
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np

import torch
from pytorch_lightning import Trainer

from classifier.utils.logconf import logging
from classifier.utils.utils import enumerate_with_estimate
from classifier.models import get_model
from classifier.runner import LitModule
from classifier.trainer import get_trainer
from classifier.datasets import get_dataloader
from classifier.losses import LossBuilder


log = logging.getLogger(__name__)


class Predictor:
    def __init__(self, config: EasyDict, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.df = pd.read_csv(config.data.folds_df)

        self.transforms = False

        self.runner = self.init_runner()
        self.model = self.runner.model
        self.model.eval()
        self.model.float()
        self.model.cuda()

    def init_runner(self):
        runner = LitModule(self.config)
        ckpt = torch.load(self.checkpoint_path)
        runner.load_state_dict(ckpt['state_dict'])

        return runner

    def init_train_dl(self):
        return get_dataloader(self.config, self.transforms, 'train')

    def init_val_dl(self):
        return get_dataloader(self.config, self.transforms, 'valid')

    def run(self):
        val_dl = self.init_val_dl()
        val_study_preds = self.do_prediction(val_dl)

        all_preds = {**val_study_preds}

        preds_df = self.organize_preds_into_df(all_preds)

        return preds_df

    def do_prediction(self, dl):
        study_preds = defaultdict(list)

        batch_iter = enumerate_with_estimate(
            dl,
            f"Predicting on {len(dl.dataset)} samples",
            start_ndx=dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            batch_preds_dict = self.predict(batch_tup)

            for study_id, study_pred in batch_preds_dict.items():
                study_preds[study_id].append(study_pred)

        return study_preds

    def predict(self, batch_tup):
        inputs, masks, labels, study_id_list = batch_tup

        inputs = inputs.cuda()
        masks = masks.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            logits = self.model(inputs).detach()
            logits += self.model(inputs.flip(2)).detach()
            logits += self.model(inputs.flip(3)).detach()
            logits += self.model(inputs.flip(2).flip(3)).detach()
            inputs = inputs.transpose(2, 3)
            logits += self.model(inputs.flip(2)).detach()
            logits += self.model(inputs.flip(3)).detach()
            logits += self.model(inputs.flip(2).flip(3)).detach()
            logits /= 8

            loss = self.runner.cls_loss_func(logits, labels).unsqueeze(dim=-1).cpu().detach().numpy()

            probability_arr = torch.nn.functional.softmax(logits, dim=-1).cpu().detach().numpy()
            preds_arr = np.hstack([probability_arr, loss])

        batch_preds_dict = dict(zip(study_id_list, preds_arr))

        return batch_preds_dict

    def organize_preds_into_df(self, preds_dict):
        preds_list = []

        for study_id, study_preds in preds_dict.items():
            study_pred = np.mean(study_preds, axis=0)
            preds_list.append([study_id] + study_pred.tolist())

        preds_df = pd.DataFrame(preds_list)
        preds_df.columns = ['study_id', 'negative', 'typical', 'indeterminate', 'atypical', 'loss']
        preds_df['label'] = preds_df['study_id'].apply(
            lambda x: self.df[self.df['study_id'] == x]['label'].values[0]
        )
        preds_df['pred_label'] = np.argmax(
            preds_df[['negative', 'typical',
                      'indeterminate', 'atypical']].values, axis=1
        )
        preds_df['fold'] = preds_df['study_id'].apply(
            lambda x: self.df[self.df['study_id'] == x]['fold'].values[0]
        )

        return preds_df

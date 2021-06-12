from easydict import EasyDict
from collections import defaultdict

import pandas as pd

import torch

from classifier.utils.logconf import logging
from classifier.utils.utils import enumerate_with_estimate
from classifier.models import get_model
from classifier.datasets import get_dataloader


log = logging.getLogger(__name__)


class Predictor:
    def __init__(self, config: EasyDict, checkpoint_path: str, output_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.df = pd.read_csv(config.data.folds_df)
        self.output_path = output_path

        self.transforms = False

        self.device = self.config.device
        self.model = self.init_model()

    def init_model(self):
        model = get_model(self.config)
        log.info("Using CUDA; current_device: {}.".format(torch.cuda.current_device()))
        if self.config.multi_gpu:
            model = torch.nn.DataParallel(model)
        model = model.to(self.device)
        self.load_checkpoint(model, self.checkpoint_path)
        model.float()
        model.eval()

        return model

    def load_checkpoint(self, model, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=self.config.device)

            if self.config.multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            best_score = checkpoint['best_score']

            print('Loaded model from checkpoint: ', checkpoint_path)
            print('Best score: ', best_score)
            print('*' * 50)

        except Exception as e:
            print('Exception loading checkpoint: ', e)
            raise

    def init_train_dl(self):
        return get_dataloader(self.config, self.transforms, 'train')

    def init_val_dl(self):
        return get_dataloader(self.config, self.transforms, 'valid')

    def run(self):
        train_dl = self.init_train_dl()
        val_dl = self.init_val_dl()

        trn_study_preds = self.do_prediction(train_dl)
        val_study_preds = self.do_prediction(val_dl)

        all_preds = {**trn_study_preds, **val_study_preds}

        preds_df = self.organize_preds_into_df(all_preds)

        preds_df.to_csv(self.output_path, index=False)

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
        input_t, label_t, study_id_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)

        with torch.no_grad():
            logits_g = self.model(input_g)
            probability_arr = torch.nn.functional.softmax(logits_g, dim=-1).cpu().detach().numpy()

        batch_preds_dict = dict(zip(study_id_list, probability_arr))

        return batch_preds_dict

    def organize_preds_into_df(self, preds_dict):
        preds_list = []

        for study_id, study_preds in preds_dict.items():
            for study_pred in study_preds:
                for label in range(4):
                    preds_list.append([study_id, label, study_pred[label], 0, 1, 0, 1])

        preds_df = pd.DataFrame(preds_list)
        preds_df.columns = ['study_id', 'label', 'conf', 'xmin', 'xmax', 'ymin', 'ymax']

        return preds_df
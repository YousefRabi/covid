from __future__ import print_function, division
import torch
import torch.nn as nn

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


class LossBuilder:

    def __init__(self, config):
        self.config = config

    def get_loss(self):
        return getattr(self, self.config.loss.name)()

    def BCE(self):
        def loss_fn(logits, targets):
            criterion = nn.BCEWithLogitsLoss(reduction=self.config.loss.params.reduction)
            return criterion(logits, targets.float())
        return loss_fn

    def SmoothBCE(self):
        def loss_fn(logits, targets):
            pos_weight = torch.tensor(self.config.loss.params.pos_weight)
            label_smoothing = self.config.loss.params.label_smoothing

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            targets = targets.float() * (1 - label_smoothing) + 0.5 * label_smoothing
            logits = logits.flatten()
            return criterion(logits, targets)

        return loss_fn

    def CrossEntropy(self):
        def loss_fn(logits, targets):
            criterion = nn.CrossEntropyLoss(reduction=self.config.loss.params.reduction)
            return criterion(logits, targets)
        return loss_fn

    def MSE(self):
        def loss_fn(logits, targets):
            logits, targets = logits, targets.float()
            criterion = nn.MSELoss(reduction=self.config.loss.params.reduction)
            return criterion(logits, targets)
        return loss_fn

    def Kappa(self):
        def loss_fn(logits, targets):
            logits = (logits - self.config.loss.params.labels_mean).flatten()
            targets = (targets - self.config.loss.params.labels_mean).flatten()
            loss = (1 - (2 * torch.dot(logits, targets) /
                    (torch.dot(targets, targets) + torch.dot(logits, logits))))
            return loss
        return loss_fn

    def attention_mining_loss(self):
        def loss_fn(logits_am, labels):
            '''Labels have to be one-hot encoded'''
            # Eq 5
            loss_am = torch.sigmoid(logits_am) * labels
            loss_am = loss_am.sum(dim=1)
            return loss_am
        return loss_fn

    def extra_supervision_loss(self):
        def loss_fn(attention_maps, masks):
            # Eq 7
            min_val = attention_maps.amin(dim=[1, 2, 3], keepdim=True)
            max_val = attention_maps.amax(dim=[1, 2, 3], keepdim=True)
            attention_maps = (attention_maps - min_val) / (max_val - min_val + 1e-6)
            criterion = nn.BCEWithLogitsLoss(reduction=self.config.loss.params.reduction)
            loss_ext = criterion(attention_maps, masks)
            return loss_ext
        return loss_fn

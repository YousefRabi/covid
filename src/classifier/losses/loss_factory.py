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

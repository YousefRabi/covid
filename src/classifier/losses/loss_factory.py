from __future__ import print_function, division
import torch
import torch.nn as nn

from classifier.utils.logconf import logging
from .lovasz_losses import lovasz_hinge


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

    def BiTemperedSoftmax(self):
        def loss_fn(logits, targets):
            loss = bi_tempered_logistic_loss(logits, targets,
                                             self.config.loss.params.t1,
                                             self.config.loss.params.t2,
                                             self.config.loss.params.label_smoothing,
                                             self.config.loss.params.num_iters)
            return loss
        return loss_fn


def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None  # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5):

    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """

    if label_smoothing > 0.0:
        num_classes = labels.shape[-1]
        labels = (1 - num_classes / (num_classes - 1) * label_smoothing) * labels + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (log_t(labels + 1e-10, t1) - log_t(probabilities, t1)) * labels
    temp2 = (1 / (2 - t1)) * (torch.pow(labels, 2 - t1) - torch.pow(probabilities, 2 - t1))
    loss_values = temp1 - temp2

    return torch.sum(loss_values, dim=-1)

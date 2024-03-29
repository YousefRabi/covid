import pretrainedmodels
import torch
import torch.nn as nn
import torchvision
import timm

import numpy as np


class ClassificationModel(nn.Module):
    def __init__(self, encoder_name, num_classes, pretrained=True):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        if 'se_resnext' in self.encoder_name:
            self.net = getattr(pretrainedmodels, self.encoder_name)(
                pretrained='imagenet')
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.last_linear = nn.Linear(
                self.net.last_linear.in_features, self.num_classes)

        elif 'resnet' in self.encoder_name:
            self.net = getattr(torchvision.models,
                               self.encoder_name)(pretrained=True)
            self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.num_classes)

        elif 'efficientnet' in self.encoder_name:
            self.net = timm.create_model(self.encoder_name, pretrained)
            self.net.classifier = nn.Linear(
                self.net.classifier.in_features, self.num_classes)

        else:
            raise NotImplementedError(f'Model name {self.encoder_name} '
                                      'not implemented. Please choose '
                                      'from se_resnext, resnet, or efficientnet variants.')

    def fresh_params(self):
        if 'se_resnext' in self.encoder_name:
            return self.net.last_linear.parameters()
        elif 'resnet' in self.encoder_name:
            return self.net.fc.parameters()
        elif 'efficientnet' in self.encoder_name:
            return self.net._fc.parameters()

    def base_params(self):
        params = []

        if 'se_resnext' in self.encoder_name:
            fc_name = 'last_linear'
        elif 'resnet' in self.encoder_name:
            fc_name = 'fc'
        elif 'efficientnet' in self.encoder_name:
            fc_name = '_fc'
        for name, param in self.net.named_parameters():
            if fc_name not in name:
                params.append(param)
        return params

    def forward(self, x):
        return self.net(x)


class MultiClsModels:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x)[:, -4:])
        res = torch.stack(res)
        return torch.mean(res, dim=0)


class MultiClsEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def __call__(self, x):
        x = x.cuda()
        with torch.no_grad():
            for i, fold_models in enumerate(self.models):
                fold_preds = []
                for model in fold_models:
                    logits = model(x)
                    probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
                    fold_preds.append(probabilities.tolist())
                fold_preds = np.mean(fold_preds, axis=0)
                if i == 0:
                    ensemble = fold_preds
                else:
                    ensemble = self.weights[i] * fold_preds + (1 - self.weights[i]) * ensemble
        return ensemble

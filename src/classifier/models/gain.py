import torch
from torch import nn
from torch.nn import functional as F

import timm


LOGITS = 'logits'
ATTENTION_MAPS = 'attention_maps'
I_STAR = 'I_star'
LOGITS_AM = 'logits_am'


class GAIN(nn.Module):
    def __init__(self,
                 efnet_encoder_name: str,
                 pretrained: bool,
                 num_classes: int,
                 gradient_layer_name: str,
                 alpha: float = 1,
                 omega: float = 10,
                 sigma: float = 0.5):
        super().__init__()
        self.model = timm.create_model(efnet_encoder_name, pretrained)
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes)

        self._register_hooks(gradient_layer_name)

        self.alpha = alpha
        self.omega = omega
        self.sigma = sigma

    def _register_hooks(self, layer_name):
        '''This wires up a hook that stores both the activation and
        gradient of the conv layer we are interested in as attributes'''
        def forward_hook(module, input_, output_):
            self._last_activation = output_

        def backward_hook(module, grad_in, grad_out):
            self._last_grad = grad_out[0]

        # locate the layer that we are concerned about
        gradient_layer_found = False
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError(f'Gradient layer {layer_name} not found in the internal model')

    def _attention_map_forward(self, images, labels, mode_str):
        logits = self.model(images)

        if mode_str == 'val':
            return logits

        grad_target = (logits * labels).sum()
        grad_target.backward(retain_graph=True)
        # grad_target is not the gradient we want to backpropagate, it's only to calculate w_c below
        # this is why we zero here
        self.model.zero_grad()

        # Eq 1
        w_c = F.adaptive_avg_pool2d(self._last_grad, 1)  # Importance weights of class c

        # Eq 2
        weights = self._last_activation
        A_c = torch.mul(weights, w_c).sum(dim=1, keepdim=True)
        A_c = F.upsample_bilinear(A_c, size=images.shape[2:])

        return logits, A_c

    def _mask_images(self, A_c, images):
        A_c_min = torch.amin(A_c, dim=[1, 2, 3], keepdim=True)
        A_c_max = torch.amax(A_c, dim=[1, 2, 3], keepdim=True)
        scaled_A_c = (A_c - A_c_min) / (A_c_max - A_c_min)

        # Eq 4
        masks = torch.sigmoid(self.omega * (scaled_A_c - self.sigma))
        # Eq 3
        masked_images = images - (images * masks)
        return masked_images

    def forward(self, images, labels, mode_str):
        '''labels have to be one-hot encoded'''
        results = {}

        labels = torch.nn.functional.one_hot(labels, num_classes=4)

        if mode_str == 'val':
            logits = self._attention_map_forward(images, labels, mode_str)
            results[LOGITS] = logits
            return results

        logits, A_c = self._attention_map_forward(images, labels, mode_str)

        # Eq 3, 4
        I_star = self._mask_images(A_c, images)

        logits_am = self.model(I_star)

        results[LOGITS] = logits
        results[ATTENTION_MAPS] = A_c
        results[I_STAR] = I_star
        results[LOGITS_AM] = logits_am

        return results

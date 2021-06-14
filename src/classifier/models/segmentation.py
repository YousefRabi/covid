import torch
import torch.nn as nn
from torch.nn import functional as F
import timm


class SegmentationModel(nn.Module):
    def __init__(self, encoder_name, num_classes, pretrained=True):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        net = timm.create_model(self.encoder_name, pretrained)
        self.b0 = nn.Sequential(
            net.conv_stem,
            net.bn1,
            net.act,
        )
        self.b1 = net.blocks[0]
        self.b2 = net.blocks[1]
        self.b3 = net.blocks[2]
        self.b4 = net.blocks[3]
        self.b5 = net.blocks[4]
        self.b6 = net.blocks[5]
        self.b7 = net.blocks[6]
        self.b8 = nn.Sequential(
            net.conv_head,  # 384, 1536
            net.bn2,
            net.act2,
        )

        self.logit = nn.Linear(1536, self.num_classes)
        self.mask = nn.Sequential(
            nn.Conv2d(136, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        def forward(self, x):
            x = self.b0(x)
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4(x)
            x = self.b5(x)

            mask = self.mask(x)

            x = self.b6(x)
            x = self.b7(x)
            x = self.b8(x)
            x = F.adaptive_avg_pool2d(x, 1)
            logit = self.logit(x)

            return logit, mask


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

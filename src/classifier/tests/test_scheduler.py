import torch

import matplotlib.pyplot as plt

from classifier.schedulers import SchedulerBuilder
from classifier.optimizers import get_optimizer
from classifier.transforms import get_transforms
from classifier.datasets import get_dataloader
from classifier.utils.config import load_config


config = load_config('/home/yousef/deep-learning/kaggle/covid/src/classifier/config/83/fold-0.yml')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 2)

    def forward(self, x):
        return self.linear(x)


model = Model()
optimizer = get_optimizer(model.parameters(), config)
transforms = get_transforms(config.transforms.train)
dataloader = get_dataloader(config, transforms, 'train')
dataset_length = len(dataloader.dataset)
scheduler_builder = SchedulerBuilder(optimizer, config, dataset_length)
scheduler = scheduler_builder.get_scheduler()

lrs = []

for i in range(config.train.num_epochs * dataset_length // config.train.batch_size):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    lrs.append(lr)


fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(lrs)
plt.savefig('runs/83/lr_cosineannealing.png')

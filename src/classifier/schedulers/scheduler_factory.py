import math

from torch.optim.lr_scheduler import (_LRScheduler, ReduceLROnPlateau, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, ExponentialLR)
from torch.optim.lr_scheduler import OneCycleLR
from warmup_scheduler import GradualWarmupScheduler

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class SchedulerBuilder:
    def __init__(self, optimizer, config, training_set_size=None):
        self.optimizer = optimizer
        self.config = config
        self.training_set_size = training_set_size

    def get_scheduler(self):
        if not self.config.scheduler.name:
            return None

        scheduler = getattr(self, self.config.scheduler.name)()

        if self.config.scheduler.warmup.apply:
            scheduler = self.warmup(scheduler)

        return scheduler

    def warmup(self, after_scheduler):
        scheduler = GradualWarmupSchedulerV2(self.optimizer,
                                             multiplier=self.config.scheduler.warmup.multiplier,
                                             total_epoch=self.config.scheduler.warmup.epochs,
                                             after_scheduler=after_scheduler)
        return scheduler

    def plateau(self):
        scheduler = ReduceLROnPlateau(self.optimizer,
                                      **self.config.scheduler.params)
        return scheduler

    def cosine(self):
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.scheduler_step,
                                      eta_min=self.config.scheduler.params.min_lr)
        return scheduler

    def warmcosine(self):
        scheduler = CosineAnnealingWarmRestarts(self.optimizer,
                                                T_0=self.scheduler_step,
                                                eta_min=self.config.scheduler.params.min_lr)
        return scheduler

    def halfcosine(self):
        scheduler = HalfCosineAnnealingLR(self.optimizer,
                                          T_max=self.scheduler_step,
                                          last_epoch=-1)
        return scheduler

    def onecycle(self):
        scheduler = OneCycleLR(self.optimizer,
                               total_steps=self.config.train.max_num_steps,
                               max_lr=self.config.scheduler.params.max_lr,
                               div_factor=self.config.scheduler.params.div_factor,
                               pct_start=self.config.scheduler.params.pct_start,
                               final_div_factor=self.config.scheduler.params.final_div_factor)
        return scheduler

    def exponential(self):
        scheduler = ExponentialLR(self.optimizer,
                                  gamma=self.config.scheduler.params.gamma)
        return scheduler

    @property
    def total_iterations(self):
        total_iterations = math.ceil(self.training_set_size / self.config.train.batch_size) *\
            self.config.train.num_epochs

        return total_iterations

    @property
    def scheduler_step(self):
        scheduler_step = (self.total_iterations // self.config.train.snapshots
                          if not hasattr(self.config.scheduler.params, 'n_steps')
                          else self.config.scheduler.params.n_steps)
        return scheduler_step

    @property
    def steps_per_epoch(self):
        steps_per_epoch = self.total_iterations // self.config.train.num_epochs
        return steps_per_epoch


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                    for base_lr in self.base_lrs]


class HalfCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.T_max = T_max
        super(HalfCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % (2 * self.T_max) < self.T_max:
            cos_unit = 0.5 * \
                (math.cos(math.pi * self.last_epoch / self.T_max) - 1)
        else:
            cos_unit = 0.5 * \
                (math.cos(math.pi * (self.last_epoch / self.T_max - 1)) - 1)

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * 1.0e-4
            range = math.log10(base_lr - math.log10(min_lr))
            lrs.append(10 ** (math.log10(base_lr) + range * cos_unit))
        return lrs

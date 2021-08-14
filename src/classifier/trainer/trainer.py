from pathlib import Path
import datetime

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from classifier.callbacks import get_callbacks
from classifier.utils.logconf import logging

log = logging.getLogger(__name__)


def get_trainer(config):
    tb_logger = pl_loggers.TensorBoardLogger(config.work_dir)

    callbacks = get_callbacks(config)

    accelerator = None if len(config.gpus) == 1 else 'ddp'

    trainer = Trainer(precision=16,
                      gpus=config.gpus,
                      logger=tb_logger,
                      default_root_dir=config.work_dir,
                      max_epochs=config.train.num_epochs,
                      callbacks=callbacks,
                      accelerator=accelerator)

    return trainer


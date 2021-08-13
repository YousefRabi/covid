from pathlib import Path
import datetime

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers


def get_trainer(config):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    model_path = Path('runs') / str(config.experiment_version) / time_str / f'fold-{config.data.idx_fold}'

    tb_logger = pl_loggers.TensorBoardLogger(model_path)

    trainer = Trainer(precision=16,
                      gpus=config.gpus,
                      logger=tb_logger,
                      default_root_dir=model_path,
                      max_epochs=config.train.num_epochs)

    return trainer


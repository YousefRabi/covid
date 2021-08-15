from argparse import ArgumentParser
from pathlib import Path
import datetime

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import seed_everything

from classifier.utils.config import load_config, save_config
from classifier.utils.logconf import logging
from classifier.runner import LitModule
from classifier.trainer import get_trainer


log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    return args


def main():
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    args = parse_args()
    config_path = Path(args.config)

    config = load_config(config_path)
    config_work_dir = Path('runs') / config_path.parent.stem / time_str / f'fold-{config.data.idx_fold}'

    log.info(f'Experiment version: {config_path.parent}')

    seed_everything(config.seed)

    config.work_dir = config_work_dir
    config.experiment_version = config_path.parent.stem
    config.exp_name = config.experiment_version + '_' + time_str
    Path(config.work_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    save_config(config, config.work_dir / 'config.yml')

    log.info(f'Fold: {config.data.idx_fold}/{config.data.num_folds}')

    trainer = get_trainer(config)
    pl_runner = LitModule(config)

    log.info(f'callbacks: {trainer.callbacks}')

    trainer.fit(pl_runner)


if __name__ == '__main__':
    main()

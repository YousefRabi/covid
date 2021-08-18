import os
from argparse import ArgumentParser
from pathlib import Path
import datetime

import torch

from classifier.runner import Runner
from classifier.utils.config import load_config, save_config
from classifier.utils.logconf import logging
from classifier.utils.utils import fix_seed


log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    return args


def init_process(rank, world_size, config_path, main_fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    main_fn(config_path, rank)


def cleanup_ddp():
    torch.distributed.destroy_process_group()


def main(config_path, rank=None):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    config_path = Path(config_path)

    config = load_config(config_path)
    config_work_dir = Path('trained-models') / config_path.parent.stem / time_str / f'fold-{config.data.idx_fold}'

    fix_seed(config.seed)

    log.info(f'Experiment version: {config_path.parent}')

    config.work_dir = config_work_dir
    config.experiment_version = config_path.parent.stem
    config.exp_name = config.experiment_version + '_' + time_str
    Path(config.work_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    save_config(config, config.work_dir / 'config.yml')

    log.info(f'Fold: {config.data.idx_fold}/{config.data.num_folds}')

    training_app = Runner(config, args.local_rank)
    training_app.run()


if __name__ == '__main__':

    args = parse_args()
    num_cuda_devices = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    if num_cuda_devices > 1:
        world_size = num_cuda_devices
        torch.multiprocessing.spawn(init_process,
                                    args=(args.rank, world_size, args.config_path),
                                    nproces=world_size,
                                    join=True)

    else:
        main(args.config_path)

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
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def init_process(rank, world_size, config_path, main_fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

    main_fn(config_path, rank, world_size)


def cleanup_ddp():
    torch.distributed.destroy_process_group()


def main(config_path, rank=0, world_size=1):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    config_path = Path(config_path)

    config = load_config(config_path)
    config.rank = rank
    config.world_size = world_size

    config_work_dir = Path('trained-models') / config_path.parent.stem / time_str / f'fold-{config.data.idx_fold}'

    fix_seed(config.seed)

    config.work_dir = config_work_dir
    config.experiment_version = config_path.parent.stem
    config.exp_name = config.experiment_version + '_' + time_str
    Path(config.work_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    save_config(config, config.work_dir / 'config.yml')

    if rank == 0:
        log.info(f'Experiment version: {config_path.parent}')
        log.info(f'Fold: {config.data.idx_fold}/{config.data.num_folds}')

    training_app = Runner(config)
    training_app.run()


if __name__ == '__main__':

    args = parse_args()
    world_size = torch.cuda.device_count()

    if world_size > 1:
        torch.multiprocessing.spawn(init_process,
                                    args=(world_size, args.config, main),
                                    nprocs=world_size,
                                    join=True)

    else:
        main(args.config)

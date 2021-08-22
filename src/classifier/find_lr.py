import os
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.dataset_factory import get_dataloader
from transforms.transform_factory import get_transforms
from transforms import Mixup
from models.model_factory import get_model
from losses.loss_factory import LossBuilder
from optimizers.optimizer_factory import get_optimizer
from utils.config import load_config
from utils.utils import fix_seed, enumerate_with_estimate
from utils.logconf import logging


log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_lr', default=0.01, type=float)
    args = parser.parse_args()
    return args


def init_process(rank, world_size, config_path, max_lr, main_fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

    main_fn(config_path, max_lr, rank, world_size)


def main(config_path, max_lr, rank, world_size=1):
    config = load_config(config_path)
    config.work_dir = Path('trained-models') / Path(config_path).parent.stem
    config.world_size = world_size
    config.device = rank
    if rank == 0:
        log.info(f'working directory: {config.work_dir}')

        Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    fix_seed(config.seed)

    train_transforms = get_transforms(config.transforms.train)
    mixup_fn = Mixup(**config.mixup) if config.mixup.prob > 0.0 else None

    if rank == 0:
        log.info(f'train_transforms: {train_transforms}')
    train_loader = get_dataloader(config, train_transforms, split='train')

    device = torch.cuda('cuda') if world_size == 1 else rank
    model = get_model(config)
    model = model.to(device)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device])

    loss_builder = LossBuilder(config)
    cls_loss_func = loss_builder.get_loss()
    seg_loss_func = loss_builder.BCE()
    optimizer = get_optimizer(model.parameters(), config)
    scaler = torch.cuda.amp.GradScaler()

    find_lr(config, rank, model, optimizer, mixup_fn, cls_loss_func, seg_loss_func,
            train_loader, scaler, final_value=max_lr)


def find_lr(config, rank, model, optimizer, mixup_fn, cls_loss_func, seg_loss_func, train_loader,
            scaler, init_value=1e-8, final_value=10., beta=0.98):

    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    accumulation_steps = config.train.accumulation_steps
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    batch_iter = enumerate_with_estimate(
        train_loader,
        "E{} Training".format(1),
        rank,
        start_ndx=train_loader.num_workers,
    )

    for i, data in batch_iter:
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, masks_t, labels, study_ids = data
        inputs = inputs.to(config.device)
        masks_g = masks_t.to(config.device)
        labels = labels.to(config.device)
        optimizer.zero_grad()

        if mixup_fn is not None:
            inputs, masks_g, labels = mixup_fn(inputs, masks_g, labels)

        with autocast():
            logits, mask_pred_g = model(inputs, return_mask=True)
            cls_loss_g = cls_loss_func(
                logits,
                labels,
            )

            seg_loss_g = seg_loss_func(
                mask_pred_g,
                masks_g,
            )

        loss = cls_loss_g.mean() + config.loss.params.seg_multiplier * seg_loss_g.mean()

        if config.train.accumulation_steps != 1:
            loss /= accumulation_steps

        loss = loss.mean()
        # Do the SGD step
        scaler.scale(loss).backward()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        # Compute the smoothed loss
        if config.world_size > 1:
            loss = loss.detach().unsqueeze(dim=0)
            torch.distributed.all_reduce(loss)
            loss /= config.world_size
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 2 * best_loss and rank == 0:
            log.info('Loss exploding, stopping...')
            plot_lr_finder_graph(config, log_lrs, losses)
            return log_lrs, losses

        if smoothed_loss < best_loss or batch_num == 1 and rank == 0:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

    if rank == 0:
        plot_lr_finder_graph(config, log_lrs, losses)
    return log_lrs, losses


def plot_lr_finder_graph(config, log_lrs, losses):
    fig, ax = plt.subplots(figsize=(12, 7))
    lrs = [10**x for x in log_lrs]
    ax.plot(lrs[10:-5], losses[10:-5])
    ax.set_xscale('log')
    locmaj = ticker.LogLocator(base=10.0, subs=(1,), numticks=100)
    ax.xaxis.set_major_locator(locmaj)
    locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=100)
    ax.xaxis.set_minor_locator(locmin)
    ax.set_title('Learning Rate Finder')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')

    if config.train.accumulation_steps != 1:
        figure_name = (f'lr_finder_{config.train.batch_size}_batch_'
                       f'{config.train.accumulation_steps}_accumulation_steps.jpg')
    else:
        figure_name = f'lr_finder_{config.train.batch_size}_batch.jpg'

    plt.savefig(Path(config.work_dir) / figure_name)


if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()

    if world_size > 1:
        torch.multiprocessing.spawn(init_process,
                                    args=(world_size, args.config, args.max_lr, main),
                                    nprocs=world_size,
                                    join=True)

    else:
        main(args.local_rank, args.config, args.max_lr)

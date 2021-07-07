from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
from torch.cuda.amp import autocast

from datasets.dataset_factory import get_dataloader
from transforms.transform_factory import get_transforms
from models.model_factory import get_model
from losses.loss_factory import LossBuilder
from optimizers.optimizer_factory import get_optimizer
from utils.config import load_config, save_config
from utils.utils import fix_seed


def main():
    args = parse_args()
    run(args)


def run(args):
    config = load_config(args.config_file)
    config.work_dir = Path('trained-models') / Path(args.config_file).parent.stem
    print('working directory:', config.work_dir)

    Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    fix_seed(config.seed)

    train_transforms = get_transforms(config.transforms.train)

    print('train_transforms: ', train_transforms)
    train_loader = get_dataloader(config, train_transforms, split='train')

    model = get_model(config)
    loss_builder = LossBuilder(config)
    cls_loss_func = loss_builder.get_loss()
    seg_loss_func = loss_builder.BCE()
    optimizer = get_optimizer(model.parameters(), config)
    scaler = torch.cuda.amp.GradScaler()
    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(config.device)

    find_lr(config, model, optimizer, cls_loss_func, seg_loss_func,
            train_loader, scaler, final_value=args.max_lr)


def find_lr(config, model, optimizer, cls_loss_func, seg_loss_func, train_loader,
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

    print('Training for one epoch to find best LR...')
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, masks_t, labels, study_ids = data
        inputs = inputs.cuda()
        masks_g = masks_t.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

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
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 2 * best_loss:
            print('Loss exploding, stopping...')
            plot_lr_finder_graph(config, log_lrs, losses)
            return log_lrs, losses

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--max_lr', default=0.01, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

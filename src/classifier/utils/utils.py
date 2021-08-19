import io
import time
import datetime
import random

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from .logconf import logging


log = logging.getLogger(__name__)


def confusion_matrix_to_image(confusion_matrix):
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(confusion_matrix, annot=True, linewidths=0.5,
                linecolor="red", fmt=".0f", ax=ax,
                xticklabels=['negative', 'typical', 'indeterminate', 'atypical'],
                yticklabels=['negative', 'typical', 'indeterminate', 'atypical'])

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    image = plot_to_image(fig)
    return image


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    image = plt.imread(buf)
    return image


def img2tensor(image: np.ndarray, dtype: np.dtype = np.float32):
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image.astype(dtype, copy=False))


def save_model_with_optimizer(model, optimizer, scheduler,
                              best_score, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_score': best_score,
    }, path)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enumerate_with_estimate(
    iterable,
    desc_str,
    rank,
    start_ndx=0,
    print_ndx=4,
    backoff=None,
    iter_len=None,
):
    """
    In terms of behavior, `enumerate_with_estimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iterable: `iterable` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iterable)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    if rank == 0:
        log.warning("{} ----/{}, starting".format(
            desc_str,
            iter_len,
        ))

    start_ts = time.time()
    for (current_ndx, item) in enumerate(iterable):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len - start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            if rank == 0:
                log.info("{} {:-4}/{}, done at {}, {}".format(
                    desc_str,
                    current_ndx,
                    iter_len,
                    str(done_dt).rsplit('.', 1)[0],
                    str(done_td).rsplit('.', 1)[0],
                ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    if rank == 0:
        log.warning("{} ----/{}, done at {}".format(
            desc_str,
            iter_len,
            str(datetime.datetime.now()).rsplit('.', 1)[0],
        ))

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def get_callbacks(config):
    callbacks = []

    if config.callbacks.checkpoint.apply:
        checkpoint_cb = ModelCheckpoint(dirpath=Path(config.work_dir) / 'checkpoints',
                                        filename='{epoch:02d}-map{map/val:.2f}',
                                        **config.callbacks.checkpoint.params)
        callbacks.append(checkpoint_cb)

    if config.callbacks.lr_monitor:
        lr_monitor_cb = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor_cb)

    return callbacks

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


def get_callbacks(config):
    callbacks = []

    if config.callbacks.checkpoint.apply:
        checkpoint_callback = ModelCheckpoint(dirpath=Path(config.work_dir) / 'checkpoints',
                                              filename='epoch{epoch:02d}-val_map{val/map:.2f}',
                                              **config.callbacks.checkpoint.params)
        callbacks.append(checkpoint_callback)

    return callbacks

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


def get_callbacks(config):
    callbacks = []

    if config.callbacks.checkpoint.apply:
        checkpoint_callback = ModelCheckpoint(dirpath=Path(config.work_dir) / 'checkpoints',
                                              monitor=config.callbacks.checkpoint.monitor,
                                              filename='best_model')
        callbacks.append(checkpoint_callback)

    return callbacks

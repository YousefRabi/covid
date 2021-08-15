from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint


def get_callbacks(config):
    callbacks = []

    if config.callbacks.checkpoint.apply:
        checkpoint_callback = ModelCheckpoint(dirpath=Path(config.work_dir) / 'checkpoints',
                                              filename='{epoch:02d}-{val_map:.2f}',
                                              monitor='val_map',
                                              verbose=True,
                                              mode='max')
        callbacks.append(checkpoint_callback)

    return callbacks

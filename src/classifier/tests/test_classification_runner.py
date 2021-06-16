import shutil
from pathlib import Path

import pytest

import torch

from classifier.utils.config import load_config
from classifier.runner.runner_factory import Runner


pytestmark = pytest.mark.skip(reason='Skipping classification tests as this branch uses segmentation method')


@pytest.fixture
def runner():
    try:
        shutil.rmtree('build/')
    except FileNotFoundError:
        pass

    config = load_config('src/classifier/tests/data/config.yml')
    config.work_dir = 'build/'
    config.experiment_version = 'test'
    config.exp_name = 'test'
    runner = Runner(config)
    return runner


def test_init_model(runner):
    model = runner.init_model()
    rand_input = torch.randint(low=0, high=255, size=(1, 3, 256, 256))
    rand_input = rand_input.to('cuda').float()

    output = model(rand_input)

    assert isinstance(output, torch.Tensor)
    assert 0 <= torch.argmax(output) <= 3


def test_run():
    '''Will test if pipeline is able to run for a single epoch on test_data and produce
    artifacts in the correct places (train.log and model artificats in config.work_dir).'''
    try:
        shutil.rmtree('build/')
    except FileNotFoundError:
        pass

    config = load_config('src/classifier/tests/data/config.yml')
    config.experiment_version = 'test'
    config.exp_name = 'test'
    config.train.num_epochs = 1

    runner = Runner(config)
    runner.run()

    assert (Path(config.work_dir) / 'train.log').exists()
    assert (Path(config.work_dir) / 'checkpoints' / 'latest_model.pth').exists()


def test_run_loss_decrease():
    '''Tests if training loss decreases on dataset of a single batch on the 2nd epoch.'''
    try:
        shutil.rmtree('build/')
    except FileNotFoundError:
        pass

    config = load_config('src/classifier/tests/data/config.yml')
    config.experiment_version = 'test'
    config.exp_name = 'test'
    config.train.num_epochs = 1
    runner = Runner(config)
    trn_loss_epoch_1 = runner.run()

    config.train.num_epochs = 2
    runner = Runner(config)
    trn_loss_epoch_2 = runner.run()

    assert trn_loss_epoch_1 > trn_loss_epoch_2

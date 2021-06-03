import imgaug.augmenters as iaa

from classifier.utils.config import load_config
from classifier.transforms.transform_factory import get_transforms


def test_get_transforms():
    config = load_config('src/classifier/tests/data/config.yml')

    transforms = get_transforms(config.transforms.train)

    assert isinstance(transforms, iaa.RandAugment)

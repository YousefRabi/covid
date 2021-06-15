from .classification import ClassificationModel
from .segmentation import SegmentationModel

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


def get_model(config):
    model = globals().get(config.model.arch)

    return model(**config.model.params)

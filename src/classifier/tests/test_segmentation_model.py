from timm.models.efficientnet import EfficientNet

from classifier.models.segmentation import SegmentationModel


def test_init():
    model = SegmentationModel('efficientnet_b0', num_classes=1)

    assert isinstance(model, SegmentationModel)
    assert hasattr(model, 'logit')
    assert hasattr(model, 'mask')

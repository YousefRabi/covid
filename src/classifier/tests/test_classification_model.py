from timm.models.efficientnet import EfficientNet

from classifier.models.classification import ClassificationModel


def test_init():
    model = ClassificationModel('efficientnet_b0', num_classes=1)

    assert isinstance(model, ClassificationModel)
    assert isinstance(model.net, EfficientNet)

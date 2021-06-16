import torch

from classifier.models.segmentation import SegmentationModel


def test_init():
    rand_input = torch.randint(low=0, high=255, size=(1, 3, 512, 512))
    rand_input = rand_input.to('cuda').float()

    model = SegmentationModel('efficientnet_b3', num_classes=1)
    model.to('cuda')

    assert isinstance(model, SegmentationModel)
    assert hasattr(model, 'logit')
    assert hasattr(model, 'mask')

    logit, mask = model(rand_input)

    assert isinstance(logit, torch.Tensor)
    assert 0 <= torch.argmax(logit) <= 3
    assert mask.shape == (1, 1, 32, 32)

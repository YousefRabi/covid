import numpy as np

from utils.utils import convert_yolo_boxes_to_sub_boxes


def test_convert_yolo_boxes_to_sub_boxes():
    yolo_boxes_1 = np.array([[0, 0.5, 0.5, 0.5, 0.5, 0.5]])
    yolo_boxes_2 = np.array([[0, 0.3, 0.25, 0.75, 0.3, 0.2]])
    image = np.random.randint(low=0, high=255, size=(1024, 1024, 3))

    sub_boxes_1 = convert_yolo_boxes_to_sub_boxes(yolo_boxes_1, image)
    sub_boxes_2 = convert_yolo_boxes_to_sub_boxes(yolo_boxes_2, image)

    assert sub_boxes_1[:, 0] == 0
    assert sub_boxes_1[:, 1] == 0.5
    assert sub_boxes_1[:, 2] == 256
    assert sub_boxes_1[:, 3] == 256
    assert sub_boxes_1[:, 4] == 768
    assert sub_boxes_1[:, 5] == 768

    np.testing.assert_almost_equal(sub_boxes_2[:, 2], 102.4)
    np.testing.assert_almost_equal(sub_boxes_2[:, 3], 665.6)
    np.testing.assert_almost_equal(sub_boxes_2[:, 4], 409.6)
    np.testing.assert_almost_equal(sub_boxes_2[:, 5], 870.4)

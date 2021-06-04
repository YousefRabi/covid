import pandas as pd
import numpy as np


def get_dicom_as_dict(dicom):
    res = dict()
    keys = list(dicom.keys())
    for k in keys:
        nm = dicom[k].name
        if nm == 'Pixel Data':
            continue
        val = dicom[k].value
        res[nm] = val
    return res


def get_train_test_image_sizes():
    sizes = dict()
    sizes_train = pd.read_csv('data/processed/train/image_width_height.csv')
    sizes_test = pd.read_csv('data/processed/test/image_width_height.csv')
    sizes_df = pd.concat((sizes_train, sizes_test), axis=0)
    for index, row in sizes_df.iterrows():
        sizes[row['image_id']] = (row['height'], row['width'])
    return sizes


def convert_yolo_boxes_to_sub_boxes(boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
    img_width = image.shape[1]
    img_height = image.shape[0]

    boxes[:, 2] = boxes[:, 2] * img_width - (
        (boxes[:, 4] * img_width) / 2)
    boxes[:, 3] = boxes[:, 3] * img_height - (
        (boxes[:, 5] * img_height) / 2)
    boxes[:, 4] = boxes[:, 2] + boxes[:, 4] * img_width
    boxes[:, 5] = boxes[:, 3] + boxes[:, 5] * img_height

    return boxes

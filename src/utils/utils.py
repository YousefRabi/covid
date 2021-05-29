import pandas as pd


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

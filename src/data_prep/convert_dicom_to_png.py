from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image
import pydicom


def convert_dicom_to_png(input_path: Path, output_path: Path, use_8bit=False, div=1):
    files = list(input_path.rglob('*.dcm'))
    output_path.mkdir(parents=True, exist_ok=True)
    for f in tqdm(files, total=len(files)):
        id = f.stem
        print('Go for: {}'.format(id))
        out_path = output_path / f'{id}.png'
        if out_path.exists():
            continue
        img = read_xray(f, use_8bit=use_8bit, rescale_times=div)
        print(img.shape, img.min(), img.max(), img.dtype)
        cv2.imwrite(out_path.as_posix(), img)


def read_xray(path, voi_lut=True, fix_monochrome=True, use_8bit=True, rescale_times=None):
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float64)
    if rescale_times:
        data = cv2.resize(data,
                          (data.shape[1] // rescale_times, data.shape[0] // rescale_times),
                          interpolation=cv2.INTER_CUBIC)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)

    if use_8bit is True:
        data = (data * 255).astype(np.uint8)
    else:
        data = (data * 65535).astype(np.uint16)

    return data


def resize_xray(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im


if __name__ == '__main__':
    convert_dicom_to_png(Path('data/raw/train'),
                         Path('data/processed/train/png_div_2'),
                         use_8bit=True,
                         div=2)
    convert_dicom_to_png(Path('data/raw/test'),
                         Path('data/processed/test/png_div_2'),
                         use_8bit=True,
                         div=2)

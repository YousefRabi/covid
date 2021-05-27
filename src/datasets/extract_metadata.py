from pathlib import Path
from tqdm import tqdm

import pandas as pd

import pydicom

from utils.utils import get_dicom_as_dict


def extract_meta(input_path: Path,
                 output_path: Path):
    files = list(input_path.rglob('*.dcm'))
    output_path.mkdir(parents=True, exist_ok=True)
    dt = []
    for f in tqdm(files, total=len(files)):
        print('Go for: {}'.format(f))
        id = f.stem
        dicom = pydicom.dcmread(f)
        dict1 = get_dicom_as_dict(dicom)
        dict1['image_id'] = id
        dt.append(dict1)
    s = pd.DataFrame(dt)
    s.to_csv(output_path / 'dicom_properties.csv', index=False)

    print(s.describe())


def extract_width_height(input_path: Path,
                         output_path: Path):
    files = list(input_path.rglob('*.dcm'))
    output_path.mkdir(parents=True, exist_ok=True)
    out = open(output_path / 'image_width_height.csv', 'w')
    out.write('image_id,width,height\n')
    for f in files:
        print('Go for: {}'.format(f))
        dicom = pydicom.dcmread(f)
        image = dicom.pixel_array
        height, width = image.shape
        out.write('{},{},{}\n'.format(f.stem, width, height))
    out.close()


if __name__ == '__main__':
    trn_input_path = Path('data/raw/train')
    trn_output_path = Path('data/processed/train')
    test_input_path = Path('data/raw/test')
    test_output_path = Path('data/processed/test')

    extract_meta(trn_input_path,
                 trn_output_path)
    extract_meta(test_input_path,
                 test_output_path)
    extract_width_height(trn_input_path,
                         trn_output_path)
    extract_width_height(test_input_path,
                         test_output_path)

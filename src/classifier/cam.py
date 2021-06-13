from collections import defaultdict
from pathlib import Path
import shutil
import argparse
import cv2
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         preprocess_image

from classifier.utils.config import load_config
from classifier.models.model_factory import get_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config used to train model.')
    parser.add_argument('--weights', help="Trained model's weights file.")
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--images_folder', type=str)
    parser.add_argument('--preds_csv', type=str)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_folder', type=str, default='./grad-cam-output',
                        help='Output image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def prepare_images(images_folder: Path, preds_df: pd.DataFrame, output_folder: Path, fold: int, num_classes: int):
    study_id_list = get_study_ids_for_grad_cam(preds_df, fold, num_classes)
    test_image_paths = list(Path('/home/yousef/deep-learning/kaggle/covid/data/raw/train').rglob('*.dcm'))
    study2image_dict = defaultdict(list)
    for test_image_path in test_image_paths:
        study2image_dict[test_image_path.parent.parent.stem].append(test_image_path.stem)

    images_folder = Path(images_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for study_id in study_id_list:
        image_ids = study2image_dict[study_id]
        for image_id in image_ids:
            shutil.copy(images_folder / (image_id + '.jpg'), output_folder / (image_id + '.jpg'))


def get_study_ids_for_grad_cam(df, fold, num_classes):
    study_id_list = []
    for label in range(num_classes):
        for pred_label in range(num_classes):
            study_ids = df.loc[
                (df['label'] == label) & (df['pred_label'] == pred_label) & (df['fold'] != fold), 'study_id'].values

            try:
                study_ids = np.random.choice(study_ids, size=2)
            except ValueError:
                print(f'No image ids for label {label} and pred {pred_label} for train')

            study_id_list.extend(study_ids)

            study_ids = df.loc[
                (df['label'] == label) & (df['pred_label'] == pred_label) & (df['fold'] == fold), 'study_id'].values

            try:
                study_ids = np.random.choice(study_ids, size=2)
            except ValueError:
                print(f'No image ids for label {label} and pred {pred_label} for valid')

            study_id_list.extend(study_ids)

    return study_id_list


def main():
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    output_folder = Path(args.output_folder)
    images_folder = Path(args.images_folder)
    original_images_folder = output_folder / 'original_images'
    test_image_paths = Path('/home/yousef/deep-learning/kaggle/covid/data/raw/train').rglob('*.dcm')
    image2study_dict = {
        test_image_path.stem: test_image_path.parent.parent.stem for test_image_path in test_image_paths
    }

    try:
        shutil.rmtree(original_images_folder)
    except FileNotFoundError:
        pass

    preds_df = pd.read_csv(args.preds_csv)
    prepare_images(images_folder, preds_df, original_images_folder, args.fold, args.num_classes)

    config = load_config(args.config)
    model = get_model(config)
    model.load_state_dict(torch.load(args.weights)['model_state_dict'])

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    target_layer = model.net.blocks[-1]  # This is for EfficientNets

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    input_tensor = []
    img_names = []

    for img_path in Path(output_folder / 'original_images').iterdir():
        try:
            rgb_img = cv2.imread(img_path.as_posix(), 1)[:, :, ::-1]
        except Exception as e:
            print(f'Exception {e} while reading image at {img_path.as_posix()}')
            raise

        rgb_img = cv2.resize(rgb_img, (config.data.image_resolution, config.data.image_resolution))
        rgb_img = np.float32(rgb_img) / 255
        img_t = preprocess_image(rgb_img,
                                 mean=[0, 0, 0],
                                 std=[1, 1, 1])
        input_tensor.append(img_t)
        img_names.append(img_path.name)

    input_tensor = torch.cat(input_tensor)
    print('input_tensor shape: ', input_tensor.shape)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.

    grayscale_cams = []

    for tensor in input_tensor:
        tensor = tensor.unsqueeze(0)
        grayscale_cam = cam(input_tensor=tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        grayscale_cams.extend(grayscale_cam)

    for grayscale_cam, img_t, img_name in zip(grayscale_cams, input_tensor, img_names):
        rgb_img = img_t.permute(1, 2, 0).numpy()

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        study_id = image2study_dict[img_name.split('.')[0]]
        label = preds_df.loc[preds_df['study_id'] == study_id, 'label'].values[0]
        preds = preds_df.loc[preds_df['study_id'] == study_id][
            ['negative', 'typical', 'indeterminate', 'atypical']].values[0]

        cv2.putText(
            cam_image,
            f'Label {label}',
            (10, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            0.75,
            (0, 255, 0),
            1
        )

        cv2.putText(
            cam_image,
            f'Pred {np.argmax(preds)}',
            (10, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.75,
            (0, 255, 0),
            1
        )

        probs = [round(pred, 2) for pred in preds]
        cv2.putText(
            cam_image,
            f'Probs {probs}',
            (10, 70),
            cv2.FONT_HERSHEY_COMPLEX,
            0.75,
            (0, 255, 0),
            1
        )

        cv2.imwrite((output_folder / f'{args.method}_cam_{img_name}').as_posix(), cam_image)


if __name__ == '__main__':
    main()

from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from torchvision import models

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

from classifier.utils.config import load_config
from classifier.models.model_factory import get_model
from classifier.datasets.dataset_factory import get_dataloader
from classifier.transforms.transform_factory import get_transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config used to train model.')
    parser.add_argument('--weights', help="Trained model's weights file.")
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--output_path', type=str, default='./grad-cam-output',
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


if __name__ == '__main__':
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

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

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
    target_layer = model.net._blocks[-1]  # This is for EfficientNets

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.use_cuda)

    input_tensor = []
    img_names = []

    for img_path in Path(args.image_path).iterdir():
        rgb_img = cv2.imread(img_path.as_posix(), 1)[:, :, ::-1]
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
    cam.batch_size = 32

    grayscale_cams = cam(input_tensor=input_tensor,
                         target_category=target_category,
                         aug_smooth=args.aug_smooth,
                         eigen_smooth=args.eigen_smooth)

    for grayscale_cam, img_t, img_name in zip(grayscale_cams, input_tensor, img_names):
        rgb_img = img_t.permute(1, 2, 0).numpy()

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        gb = gb_model(img_t[None, ...], target_category=target_category)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cv2.imwrite((output_path / f'{args.method}_cam_{img_name}').as_posix(), cam_image)
        cv2.imwrite((output_path / f'{args.method}_gb_{img_name}').as_posix(), gb)
        cv2.imwrite((output_path / f'{args.method}_cam_gb_{img_name}').as_posix(), cam_gb)

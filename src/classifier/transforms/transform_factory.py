from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    RandomRotate90,
    RandomCrop,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomSizedCrop,
    RandomBrightnessContrast,
    HueSaturationValue,
    ShiftScaleRotate,
    MotionBlur,
    MedianBlur,
    Blur,
    OpticalDistortion,
    GridDistortion,
    PadIfNeeded,
    IAAPiecewiseAffine,
    OneOf)
from albumentations.pytorch import ToTensorV2
import cv2
import imgaug.augmenters as iaa

from classifier.utils.logconf import logging


log = logging.getLogger(__name__)


def get_transforms(phase_config):
    if phase_config.randaugment.p > 0:
        return iaa.RandAugment(n=phase_config.randaugment.N, m=phase_config.randaugment.M)
    return standard_aug(phase_config)


def standard_aug(phase_config):
    list_transforms = []

    if phase_config.Resize.p > 0:
        list_transforms.append(
            Resize(
                phase_config.Resize.height,
                phase_config.Resize.width,
                p=phase_config.Resize.p)
        )

    if phase_config.HorizontalFlip:
        list_transforms.append(HorizontalFlip())

    if phase_config.VerticalFlip:
        list_transforms.append(VerticalFlip())

    if phase_config.Transpose:
        list_transforms.append(Transpose())

    if phase_config.RandomRotate90:
        list_transforms.append(RandomRotate90())

    if phase_config.RandomCropScale.p > 0:
        min_height = phase_config.RandomCropScale.min_height
        max_height = phase_config.RandomCropScale.max_height
        resize_height = phase_config.RandomCropScale.resize_height
        resize_width = phase_config.RandomCropScale.resize_width
        list_transforms.append(
            RandomSizedCrop(
                min_max_height=(min_height, max_height),
                height=resize_height,
                width=resize_width,
                w2h_ratio=resize_width / resize_height)
        )

    if phase_config.ShiftScaleRotate.p > 0:
        list_transforms.append(
            ShiftScaleRotate(
                shift_limit=phase_config.ShiftScaleRotate.shift_limit,
                scale_limit=phase_config.ShiftScaleRotate.scale_limit,
                rotate_limit=phase_config.ShiftScaleRotate.rotate_limit,
                border_mode=getattr(cv2, phase_config.ShiftScaleRotate.border_mode),
                value=phase_config.ShiftScaleRotate.value,
                p=phase_config.ShiftScaleRotate.p
            )
        )

    if phase_config.RandomCrop.p > 0:
        list_transforms.append(
            RandomCrop(phase_config.RandomCrop.height,
                       phase_config.RandomCrop.width, p=1)
        )
    if phase_config.Noise:
        list_transforms.append(
            OneOf([
                GaussNoise(),
                IAAAdditiveGaussianNoise(),
            ], p=0.5),
        )

    if phase_config.Gamma.p > 0:
        list_transforms.append(
            RandomGamma(gamma_limit=phase_config.Gamma.limit, p=phase_config.Gamma.p)
        )

    if phase_config.BrightnessContrast.p > 0:
        list_transforms.append(
            RandomBrightnessContrast(brightness_limit=phase_config.BrightnessContrast.brightness_limit,
                                     contrast_limit=phase_config.BrightnessContrast.contrast_limit,
                                     p=phase_config.BrightnessContrast.p)
        )

    if phase_config.HueSaturationValue.p > 0:
        list_transforms.append(
            HueSaturationValue(hue_shift_limit=phase_config.HueSaturationValue.hue_limit,
                               sat_shift_limit=phase_config.HueSaturationValue.sat_limit,
                               val_shift_limit=phase_config.HueSaturationValue.val_limit,
                               p=phase_config.HueSaturationValue.p))

    if phase_config.Blur:
        list_transforms.append(
            OneOf([
                MotionBlur(blur_limit=3, p=0.1),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2)
        )
    if phase_config.Distort:
        list_transforms.append(
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.3)
        )

    if phase_config.Cutout.p > 0:
        num_holes = phase_config.Cutout.num_holes
        hole_size = phase_config.Cutout.hole_size
        list_transforms.append(Cutout(num_holes, hole_size, p=phase_config.Cutout.p))

    if phase_config.Pad.p > 0:
        pad_height = phase_config.Resize.height + phase_config.Pad.pad_value * 2
        pad_width = phase_config.Resize.width + phase_config.Pad.pad_value * 2
        list_transforms.append(
            PadIfNeeded(
                min_height=pad_height,
                min_width=pad_width,
                border_mode=getattr(cv2, phase_config.Pad.border_mode),
                p=1
            )
        )

    if phase_config.Normalize.apply:
        list_transforms.append(
            Normalize(mean=phase_config.Normalize.mean, std=phase_config.Normalize.std, p=1),
        )

    if phase_config.ToTensor:
        list_transforms.append(
            ToTensorV2(),
        )

    if phase_config.AssertShape.p > 0:
        return iaa.Sequential([
            iaa.AssertShape((None, phase_config.AssertShape.height, phase_config.AssertShape.width, [1, 3]))
        ])

    return Compose(list_transforms) if list_transforms else False

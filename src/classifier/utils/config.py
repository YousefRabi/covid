import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    c.project_name = ''

    c.data = edict()
    c.data.sample_submission_path = 'data/sample_submission.csv'
    c.data.data_dir = 'data/train/processed/images/'
    c.data.external_data_dir = ''
    c.data.external_data_df = ''
    c.data.params = edict()

    c.model = edict()
    c.model.arch = 'CustomModel'
    c.model.params = edict()

    c.train = edict()
    c.train.task = 'cls'
    c.train.batch_size = 32
    c.train.subset = 0
    c.train.overfit_single_batch = False
    c.train.resume = 0
    c.train.num_epochs = 50
    c.train.accumulation_steps = 1
    c.train.checkpoint_path = ''

    c.test = edict()
    c.test.batch_size = 32
    c.test.cls_threshold = 0.5
    c.test.tta = edict()
    c.test.horizontal = False
    c.test.vertical = False
    c.test.vertical_horizontal = False

    c.transforms = edict()
    c.transforms.params = edict()

    c.transforms.train = edict()
    c.transforms.train.Resize = edict()
    c.transforms.train.Resize.p = 0
    c.transforms.train.Resize.height = 320
    c.transforms.train.Resize.width = 480
    c.transforms.train.HorizontalFlip = False
    c.transforms.train.VerticalFlip = False
    c.transforms.train.Transpose = False
    c.transforms.train.RandomRotate90 = False
    c.transforms.train.RandomCropScale = edict()
    c.transforms.train.RandomCropScale.p = 0
    c.transforms.train.RandomCropScale.min_height = 512
    c.transforms.train.RandomCropScale.max_height = 512
    c.transforms.train.RandomCropScale.resize_height = 512
    c.transforms.train.RandomCropScale.resize_width = 512
    c.transforms.train.Rotate90 = False
    c.transforms.train.RandomCropRotateScale = False
    c.transforms.train.Cutout = edict()
    c.transforms.train.Cutout.p = 0
    c.transforms.train.Cutout.num_holes = 0
    c.transforms.train.Cutout.hole_size = 25
    c.transforms.train.RandomCrop = edict()
    c.transforms.train.RandomCrop.p = 0
    c.transforms.train.RandomCrop.height = 320
    c.transforms.train.RandomCrop.width = 480
    c.transforms.train.BrightnessContrast = edict()
    c.transforms.train.BrightnessContrast.p = 0
    c.transforms.train.BrightnessContrast.brightness_limit = 0.2
    c.transforms.train.BrightnessContrast.contrast_limit = 0.2
    c.transforms.train.HueSaturationValue = edict()
    c.transforms.train.HueSaturationValue.p = 0
    c.transforms.train.HueSaturationValue.hue_limit = 0.2
    c.transforms.train.HueSaturationValue.sat_limit = 0.3
    c.transforms.train.HueSaturationValue.val_limit = 0.2
    c.transforms.train.Gamma = edict()
    c.transforms.train.Gamma.p = 0
    c.transforms.train.Gamma.limit = (80, 120)
    c.transforms.train.Normalize = edict()
    c.transforms.train.Normalize.apply = False
    c.transforms.train.Normalize.mean = [0.485, 0.456, 0.406]  # ImageNet statistics
    c.transforms.train.Normalize.std = [0.229, 0.224, 0.225]  # ImageNet statistics
    c.transforms.train.ToTensor = False
    c.transforms.train.Noise = False
    c.transforms.train.Blur = False
    c.transforms.train.Distort = False
    c.transforms.train.ShiftScaleRotate = edict()
    c.transforms.train.ShiftScaleRotate.p = 0
    c.transforms.train.ShiftScaleRotate.shift_limit = 0.0625
    c.transforms.train.ShiftScaleRotate.scale_limit = 0.1
    c.transforms.train.ShiftScaleRotate.rotate_limit = 45
    c.transforms.train.ShiftScaleRotate.border_mode = 'BORDER_REFLECT'
    c.transforms.train.ShiftScaleRotate.value = 0
    c.transforms.train.CutMix = False
    c.transforms.train.Mixup = False
    c.transforms.train.Pad = edict()
    c.transforms.train.Pad.p = 0
    c.transforms.train.Pad.pad_value = 0
    c.transforms.train.randaugment = edict()
    c.transforms.train.randaugment.p = 0
    c.transforms.train.randaugment.N = 0
    c.transforms.train.randaugment.M = 0
    c.transforms.train.AssertShape = edict()
    c.transforms.train.AssertShape.p = 0
    c.transforms.train.AssertShape.height = 256
    c.transforms.train.AssertShape.width = 256

    c.transforms.test = edict()
    c.transforms.test.Resize = edict()
    c.transforms.test.Resize.p = 0
    c.transforms.test.Resize.height = 320
    c.transforms.test.Resize.width = 480
    c.transforms.test.HorizontalFlip = False
    c.transforms.test.VerticalFlip = False
    c.transforms.test.Transpose = False
    c.transforms.test.RandomRotate90 = False
    c.transforms.test.RandomCropScale = edict()
    c.transforms.test.RandomCropScale.p = 0
    c.transforms.test.RandomCropScale.min_height = 512
    c.transforms.test.RandomCropScale.max_height = 512
    c.transforms.test.RandomCropScale.resize_height = 512
    c.transforms.test.RandomCropScale.resize_width = 512
    c.transforms.test.Rotate90 = False
    c.transforms.test.RandomCropRotateScale = False
    c.transforms.test.Cutout = edict()
    c.transforms.test.Cutout.p = 0
    c.transforms.test.Cutout.num_holes = 0
    c.transforms.test.Cutout.hole_size = 25
    c.transforms.test.RandomCrop = edict()
    c.transforms.test.RandomCrop.p = 0
    c.transforms.test.RandomCrop.height = 300
    c.transforms.test.RandomCrop.width = 450
    c.transforms.test.BrightnessContrast = edict()
    c.transforms.test.BrightnessContrast.p = 0
    c.transforms.test.BrightnessContrast.brightness_limit = 0.2
    c.transforms.test.BrightnessContrast.contrast_limit = 0.2
    c.transforms.test.HueSaturationValue = edict()
    c.transforms.test.HueSaturationValue.p = 0
    c.transforms.test.HueSaturationValue.hue_limit = 0.2
    c.transforms.test.HueSaturationValue.sat_limit = 0.3
    c.transforms.test.HueSaturationValue.val_limit = 0.2
    c.transforms.test.Gamma = edict()
    c.transforms.test.Gamma.p = 0
    c.transforms.test.Gamma.limit = (80, 120)
    c.transforms.test.Normalize = edict()
    c.transforms.test.Normalize.apply = False
    c.transforms.test.Normalize.mean = [0.485, 0.456, 0.406]  # ImageNet statistics
    c.transforms.test.Normalize.std = [0.229, 0.224, 0.225]  # ImageNet statistics
    c.transforms.test.ToTensor = False
    c.transforms.test.Noise = False
    c.transforms.test.Blur = False
    c.transforms.test.Distort = False
    c.transforms.test.ShiftScaleRotate = edict()
    c.transforms.test.ShiftScaleRotate.p = 0
    c.transforms.test.ShiftScaleRotate.shift_limit = 0.0625
    c.transforms.test.ShiftScaleRotate.scale_limit = 0.1
    c.transforms.test.ShiftScaleRotate.rotate_limit = 45
    c.transforms.test.CutMix = False
    c.transforms.test.Mixup = False
    c.transforms.test.Pad = edict()
    c.transforms.test.Pad.p = 0
    c.transforms.test.Pad.pad_value = 0
    c.transforms.test.randaugment = edict()
    c.transforms.test.randaugment.p = 0
    c.transforms.test.randaugment.N = 0
    c.transforms.test.randaugment.M = 0
    c.transforms.test.AssertShape = edict()
    c.transforms.test.AssertShape.p = 0
    c.transforms.test.AssertShape.height = 256
    c.transforms.test.AssertShape.width = 256

    c.loss = edict()
    c.loss.name = 'BCE'
    c.loss.params = edict()
    c.loss.params.clip_grad = 0
    c.loss.params.reduction = 'mean'
    c.loss.params.seg_multiplier = 1

    c.optimizer = edict()
    c.optimizer.name = 'Adam'
    c.optimizer.params = edict()

    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.type = 'epoch'
    c.scheduler.params = edict()
    c.scheduler.warmup = edict()
    c.scheduler.warmup.apply = False
    c.scheduler.warmup.multiplier = 1
    c.scheduler.warmup.epochs = 1

    c.device = 'cuda:0'
    c.multi_gpus = False
    c.num_workers = 2
    c.work_dir = './work_dir'
    c.experiment_version = 0
    c.seed = 0

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.Loader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as write_file:
        yaml.dump(config, write_file)

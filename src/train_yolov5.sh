set -eux pipefail

export CUDA_VISIBLE_DEVICES='0'

python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/lung_detector/dataset_fold_0.yml --device 0 --project trained-models/lung_detector_fold0_06-11/ --weights 'yolov5s.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/lung_detector/dataset_fold_1.yml --device 0 --project trained-models/lung_detector_fold1_06-11/ --weights 'yolov5s.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/lung_detector/dataset_fold_2.yml --device 0 --project trained-models/lung_detector_fold2_06-11/ --weights 'yolov5s.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/lung_detector/dataset_fold_3.yml --device 0 --project trained-models/lung_detector_fold3-06-11/ --weights 'yolov5s.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/lung_detector/dataset_fold_4.yml --device 0 --project trained-models/lung_detector_fold4-06-11/ --weights 'yolov5s.pt'

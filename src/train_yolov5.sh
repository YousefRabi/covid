set -eux pipefail

CUDA_VISIBLE_DEVICES='0'

python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/dataset_fold_0.yml --device 0 --project trained-models/03-29_yolov5_fold0/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/dataset_fold_1.yml --device 0 --project trained-models/03-29_yolov5_fold1/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/dataset_fold_2.yml --device 0 --project trained-models/03-29_yolov5_fold2/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/dataset_fold_3.yml --device 0 --project trained-models/03-29_yolov5_fold3/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 8 --epochs 80 --data yolov5/dataset_fold_4.yml --device 0 --project trained-models/03-29_yolov5_fold4/ --weights 'yolov5x.pt'

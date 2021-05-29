CUDA_VISIBLE_DEVICES='0'

python3 yolov5/train.py --img 640 --batch 128 --epochs 80 --data /home/yousef/deep-learning/kaggle/covid/data/processed/yolov5_data/train/images/fold_0.xml --device 0 --project trained-models/03-29_yolov5_fold0/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 128 --epochs 80 --data /home/yousef/deep-learning/kaggle/covid/data/processed/yolov5_data/train/images/fold_1.xml --device 0 --project trained-models/03-29_yolov5_fold1/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 128 --epochs 80 --data /home/yousef/deep-learning/kaggle/covid/data/processed/yolov5_data/train/images/fold_2.xml --device 0 --project trained-models/03-29_yolov5_fold2/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 128 --epochs 80 --data /home/yousef/deep-learning/kaggle/covid/data/processed/yolov5_data/train/images/fold_3.xml --device 0 --project trained-models/03-29_yolov5_fold3/ --weights 'yolov5x.pt'
python3 yolov5/train.py --img 640 --batch 128 --epochs 80 --data /home/yousef/deep-learning/kaggle/covid/data/processed/yolov5_data/train/images/fold_4.xml --device 0 --project trained-models/03-29_yolov5_fold4/ --weights 'yolov5x.pt'

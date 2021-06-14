set -eux pipefail

export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/19/fold-0.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/19/fold-1.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/19/fold-2.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/19/fold-3.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/19/fold-4.yml

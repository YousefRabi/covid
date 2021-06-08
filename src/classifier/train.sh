set -eux pipefail

PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/8/fold-0.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/8/fold-1.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/8/fold-2.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/8/fold-3.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/8/fold-4.yml

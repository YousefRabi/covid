set -eux pipefail

PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/12/fold-1.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/12/fold-2.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/12/fold-3.yml
PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src' python src/classifier/train.py --config src/classifier/config/12/fold-4.yml

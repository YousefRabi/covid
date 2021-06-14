set -eux pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src/'

python src/classifier/train.py --config src/classifier/config/14/fold-0.yml
python src/classifier/train.py --config src/classifier/config/14/fold-1.yml
python src/classifier/train.py --config src/classifier/config/14/fold-2.yml
python src/classifier/train.py --config src/classifier/config/14/fold-3.yml
python src/classifier/train.py --config src/classifier/config/14/fold-4.yml

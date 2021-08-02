set -eux pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/yousef/deep-learning/kaggle/covid-exp-152/src/'

python src/classifier/train.py --config src/classifier/config/152/fold-0.yml
python src/classifier/train.py --config src/classifier/config/152/fold-1.yml
python src/classifier/train.py --config src/classifier/config/152/fold-2.yml
python src/classifier/train.py --config src/classifier/config/152/fold-3.yml
python src/classifier/train.py --config src/classifier/config/152/fold-4.yml

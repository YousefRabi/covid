set -eux pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH='/home/yousef/deep-learning/kaggle/covid-exp-81/src/'

python src/classifier/train.py --config src/classifier/config/81/fold-0.yml
python src/classifier/train.py --config src/classifier/config/81/fold-1.yml
python src/classifier/train.py --config src/classifier/config/81/fold-2.yml
python src/classifier/train.py --config src/classifier/config/81/fold-3.yml
python src/classifier/train.py --config src/classifier/config/81/fold-4.yml

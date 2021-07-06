set -eux pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/yousef/deep-learning/kaggle/covid-exp-75/src/'

python src/classifier/train.py --config src/classifier/config/75/fold-0.yml
python src/classifier/train.py --config src/classifier/config/75/fold-1.yml
python src/classifier/train.py --config src/classifier/config/75/fold-2.yml
python src/classifier/train.py --config src/classifier/config/75/fold-3.yml
python src/classifier/train.py --config src/classifier/config/75/fold-4.yml

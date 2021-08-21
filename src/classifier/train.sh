set -eux pipefail

export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH='/home/yousef/deep-learning/kaggle/covid/src/'

python -u -m torch.distributed.launch src/classifier/train.py --config src/classifier/config/83/fold-0.yml
python -u -m torch.distributed.launch src/classifier/train.py --config src/classifier/config/83/fold-1.yml
python -u -m torch.distributed.launch src/classifier/train.py --config src/classifier/config/83/fold-2.yml
python -u -m torch.distributed.launch src/classifier/train.py --config src/classifier/config/83/fold-3.yml
python -u -m torch.distributed.launch src/classifier/train.py --config src/classifier/config/83/fold-4.yml

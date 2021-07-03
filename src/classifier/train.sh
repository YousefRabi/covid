set -eux pipefail

export CUDA_VISIBLE_DEVICES=0

python src/classifier/train.py --config src/classifier/config/23/fold-0.yml
python src/classifier/train.py --config src/classifier/config/23/fold-1.yml
python src/classifier/train.py --config src/classifier/config/23/fold-2.yml
python src/classifier/train.py --config src/classifier/config/23/fold-3.yml
python src/classifier/train.py --config src/classifier/config/23/fold-4.yml

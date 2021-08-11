export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH='src/'

python src/classifier/predict.py --config src/classifier/config/75/fold-0.yml --checkpoint_path trained-models/75/all_folds/fold-0.pth --output_path trained-models/75/fold_0_preds.csv
python src/classifier/predict.py --config src/classifier/config/75/fold-1.yml --checkpoint_path trained-models/75/all_folds/fold-1.pth --output_path trained-models/75/fold_1_preds.csv
python src/classifier/predict.py --config src/classifier/config/75/fold-2.yml --checkpoint_path trained-models/75/all_folds/fold-2.pth --output_path trained-models/75/fold_2_preds.csv
python src/classifier/predict.py --config src/classifier/config/75/fold-3.yml --checkpoint_path trained-models/75/all_folds/fold-3.pth --output_path trained-models/75/fold_3_preds.csv
python src/classifier/predict.py --config src/classifier/config/75/fold-4.yml --checkpoint_path trained-models/75/all_folds/fold-4.pth --output_path trained-models/75/fold_4_preds.csv

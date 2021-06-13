from argparse import ArgumentParser

from classifier.utils.config import load_config
from classifier.predictors import Predictor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = load_config(args.config)
    predictor = Predictor(config, args.checkpoint_path, args.output_path)
    predictor.run()


if __name__ == '__main__':
    main()

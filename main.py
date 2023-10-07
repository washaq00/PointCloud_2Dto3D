import argparse
from pathlib import Path
import h5py
from configs.config_utils import CONFIG

batch_size = 20
img_width, img_height, img_channels = 28,28,1
no_classes
def load_data():

def parse_args():
    """User-friendly command lines"""
    parser = argparse.ArgumentParser('Total 3D Understanding.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser


def main():
    parser = parse_args()
    cfg = CONFIG(parser)
    cfg.log_string('Loading configurations.')
    cfg.log_string(cfg.config)

    data_path = Path("data/")
    images_path = data_path / "model1"




if __name__ == '__main__':
    main()


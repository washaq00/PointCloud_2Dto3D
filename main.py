# import argparse
# import requests
# from torchvision.transforms import Lambda
# from pathlib import Path
# from configs.config_utils import CONFIG
import pandas as pd
from CustomDataset import CustomImageDataset
from myfunctions import print_image
from torchvision import transforms


# def parse_args():
#     """User-friendly command lines"""
#     parser = argparse.ArgumentParser('Total 3D Understanding.')
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--test', action='store_true')
#     return parser


def main():

    data_transform = transforms.Compose([
                        transforms.Resize(size=(256, 256)),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ToTensor()])

    ds = CustomImageDataset(
        annotations_file=pd.read_csv('data/data.csv', sep=';'),
        img_dir="data/images/",
        transform=data_transform,
        target_transform=None
        # Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

    img, label = ds.__getitem__(4)
    print_image(img)


if __name__ == '__main__':
    main()


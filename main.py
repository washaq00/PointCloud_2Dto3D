import argparse
import requests
from torchvision.transforms import ToTensor, Lambda
from pathlib import Path
import pandas as pd
from CustomDataset import CustomImageDataset
# from configs.config_utils import CONFIG
from myfunctions import print_image
from torchvision import transforms


# batch_size = 16
# img_width, img_height, img_channels = 28,28,1
# no_classes = 4

# def parse_args():
#     """User-friendly command lines"""
#     parser = argparse.ArgumentParser('Total 3D Understanding.')
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--test', action='store_true')
#     return parser


def main():

    # input_path = '../input'
    # url = "https://example.com/image.jpg"
    # response = requests.get(url)
    # with open("image.jpg", "wb") as f:
    #     f.write(response.content)

    data_transform = transforms.Compose([
                        transforms.Resize(size=(256, 256)),
                        transforms.RandomVertical Flip(p=0.5),
                        transforms.ToTensor()])

    ds = CustomImageDataset(
        annotations_file=pd.read_csv('data/data.csv', sep=';'),
        img_dir="data/images/",
        transform=data_transform,
        target_transform= None #Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    )

    img, label = ds.__getitem__(4)
    print_image(img)

    # parser = parse_args()
    # cfg = CONFIG(parser)
    # cfg.log_string('Loading configurations.')
    # cfg.log_string(cfg.config)

if __name__ == '__main__':
    main()


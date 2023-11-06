# import argparse
# import requests
# from torchvision.transforms import Lambda
# from pathlib import Path
# from configs.config_utils import CONFIG
import os

import torch.cuda

from CustomDataset import find_classes, CustomImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader

from myfunctions import print_image


# def parse_args():
#     """User-friendly command lines"""
#     parser = argparse.ArgumentParser('Total 3D Understanding.')
#     parser.add_argument('--train', action='store_true')
#     parser.add_argument('--test', action='store_true')
#     return parser


def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if Path("helper_functions.py").is_file():
    #   print("helper_functions.py already exists, skipping download")
    # else:
    #   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    #   with open("helper_functions.py", "wb") as f:
    #     f.write(request.content)

    BATCH_SIZE = 16
    NUM_WORKERS = 2

    # implement Data Augmentation and Image Resizing
    train_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()])

    train_dir = os.path.join(os.getcwd(), "data/train/")
    print(find_classes(train_dir))
    # test_dir = os.path.join(os.getcwd(), "data/test/")

    train_data = CustomImageDataset(img_dir=train_dir,  # target folder of images
                                    transform=train_transform)  # transforms to perform on labels (if necessary)

    # test_data = CustomImageDataset(img_dir=test_dir,
    #                                transform=test_transform)

    print(f"test data {train_data}")
    class_names = train_data.classes
    print(class_names)
    img, label = train_data[6][0], train_data[6][1]
    print_image(img)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  shuffle=True)

    # test_dataloader = DataLoader(dataset=test_data,
    #                              batch_size=BATCH_SIZE,
    #                              num_workers=NUM_WORKERS,
    #                              shuffle=False)  # don't usually need to shuffle testing data


print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))

if __name__ == '__main__':
    main()


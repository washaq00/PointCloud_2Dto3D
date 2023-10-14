import os
from PIL import Image
from torch.utils.data import Dataset
import pathlib
from typing import Tuple, Dict, List #typical list is not


def find_classes(dir: str) -> Tuple[List[str],Dict[str, int]]:
    #get the class names with scanning directory (it checkes if there are any subfolder)
    classes = sorted(entry.name for entry in os.scandir(dir) if entry.is_dir())

    #show the error
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes")

    #creating a Dict with name and id
    classes_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, classes_to_id


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform =None):
        self.classes, self.classes_idx = find_classes(img_dir)
        self.paths = list(pathlib.Path(img_dir).glob("*/*.png"))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.classes_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, class_idx


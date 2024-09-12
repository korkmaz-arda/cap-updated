# The code is sourced from: 
# https://gitlab.jsc.fz-juelich.de/kiste/vissl/-/blob/c2aa96d0569caea4be264b2132d0931d2599e1ed/extra_scripts/datasets/create_food101_data_files.py
# It is also featured in an old version of the original VISSL repo (https://github.com/facebookresearch/vissl)

import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


class Food101:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """
    META_FOLDER = "meta"
    IMAGE_FOLDER = "images"
    IMAGE_EXT = ".jpg"

    def __init__(self, input_path: str, output_path: str, split: str):
        self.input_path = input_path
        self.output_path = output_path
        self.split = split
        self.class_file = os.path.join(self.input_path, self.META_FOLDER, "classes.txt")
        self.split_path = os.path.join(
            self.input_path, self.META_FOLDER, split.lower() + ".txt"
        )
        self.IMAGE_FOLDER = os.path.join(self.input_path, self.IMAGE_FOLDER)
        with open(self.class_file, "r") as f:
            self.classes = {line.strip() for line in f}

        self.targets = []
        self.images = []
        with open(self.split_path, "r") as f:
            for line in f:
                label, image_file_name = line.strip().split("/")
                assert label in self.classes, f"Invalid label: {label}"
                self.targets.append(label)
                self.images.append(
                    os.path.join(
                        # self.input_path,
                        self.IMAGE_FOLDER,
                        label,
                        image_file_name + self.IMAGE_EXT,
                    )
                )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image_name = os.path.split(image_path)[1]
        image = Image.open(image_path)
        if image.mode == "L":
            image = image.convert("RGB")
        target = self.targets[idx]
        image.save(os.path.join(self.output_path, self.split, target, image_name))
        return True


def create_food_101_disk_folder(input_path: str, output_path: str):
    for split in ["train", "test"]:
        dataset = Food101(input_path=input_path, output_path=output_path, split=split)
        for label in dataset.classes:
            os.makedirs(os.path.join(output_path, split, label), exist_ok=True)
        loader = DataLoader(
            dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0]
        )
        with tqdm(total=len(dataset)) as progress_bar:
            for _ in loader:
                progress_bar.update(1)


if __name__ == "__main__":
    input_path = './datasets/food-101'
    output_path = './datasets/food-101-splits'
    create_food_101_disk_folder(input_path=input_path, output_path=output_path)

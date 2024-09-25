import os
import shutil
import random
from tqdm import tqdm


def create_3_splits(input_path, output_path, train_percent=70, val_percent=15, test_percent=15):
    if train_percent + val_percent + test_percent != 100:
        raise ValueError("Split percentages must add up to 100.")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        for label in os.listdir(os.path.join(input_path, 'images')):
            os.makedirs(os.path.join(output_path, split, label), exist_ok=True)

    # process each food label
    for label in tqdm(os.listdir(os.path.join(input_path, 'images')), desc="Processing classes"):
        class_dir = os.path.join(input_path, 'images', label)
        images = os.listdir(class_dir)
        random.shuffle(images)  # randomize selection

        # calculate split sizes
        total_images = len(images)
        train_count = int(total_images * train_percent / 100)
        val_count = int(total_images * val_percent / 100)

        # create split lists
        train_images = sorted(images[:train_count])
        val_images = sorted(images[train_count:train_count + val_count])
        test_images = sorted(images[train_count + val_count:])

        # populate split dirs
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(output_path, 'train', label, image)
            shutil.copy(src, dst)

        for image in val_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(output_path, 'val', label, image)
            shutil.copy(src, dst)

        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(output_path, 'test', label, image)
            shutil.copy(src, dst)


if __name__ == "__main__":
    input_path = './datasets/food-101' 
    output_path = './datasets/food-101-3splits'
    create_3_splits(input_path, output_path, 75, 0, 25)
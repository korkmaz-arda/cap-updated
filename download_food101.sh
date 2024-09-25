#!/bin/bash

FOOD101_URL="http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
TAR_FILE="food-101.tar.gz"
TARGET_DIR="./datasets"

echo "Downloading Food-101 dataset..."
wget $FOOD101_URL -O $TAR_FILE

echo "Unzipping Food-101 dataset..."
tar -xzf $TAR_FILE 

if [ ! -d "$TARGET_DIR" ]; then
    echo "Target directory $TARGET_DIR does not exist. Creating it..."
    mkdir -p $TARGET_DIR
fi

echo "Moving dataset to $TARGET_DIR..."
mv food-101 $TARGET_DIR

rm $TAR_FILE
echo "Food-101 dataset downloaded, unzipped, and moved to $TARGET_DIR successfully!"

echo "Formatting the Food-101 dataset..."
python3 utils/split_food101.py

echo "Food-101 dataset formatted successfully!"
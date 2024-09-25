def check_intersections(output_path):
    """Making sure no data is shared between splits  
    """
    train_images = set()
    val_images = set()
    test_images = set()
    
    def collect_images(split_path, images_set):
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            for image in os.listdir(label_path):
                image_path = os.path.join(label, image)
                images_set.add(image_path)

    # collect images for each split
    collect_images(os.path.join(output_path, 'train'), train_images)
    collect_images(os.path.join(output_path, 'val'), val_images)
    collect_images(os.path.join(output_path, 'test'), test_images)

    # get intersections
    train_val_intersection = train_images & val_images
    train_test_intersection = train_images & test_images
    val_test_intersection = val_images & test_images

    # results
    if train_val_intersection:
        print(f"Intersection found between train and val sets: {len(train_val_intersection)} images.")
    else:
        print("No intersection between train and val sets.")

    if train_test_intersection:
        print(f"Intersection found between train and test sets: {len(train_test_intersection)} images.")
    else:
        print("No intersection between train and test sets.")

    if val_test_intersection:
        print(f"Intersection found between val and test sets: {len(val_test_intersection)} images.")
    else:
        print("No intersection between val and test sets.")

    if train_val_intersection or train_test_intersection or val_test_intersection:
        print("Error: There are intersections between the splits.")
    else:
        print("All splits are correctly separated with no intersections.")



if __name__ == "__main__":
    dataset_path = './datasets/food-101-3splits'
    check_intersections(dataset_path)
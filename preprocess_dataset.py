import argparse
import os
import shutil
from random import seed, shuffle

from utils.utils import get_dataset_path


def create_subset(num_classes, db_images_number, test_images_number):
    src_dir = "datasets/reduced_facescrub"
    if not os.path.exists(src_dir):
        print(f"The source directory {src_dir} does not exist.")
        return

    dest_dir = get_dataset_path(num_classes, db_images_number, test_images_number)
    os.makedirs(dest_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    shuffle(class_dirs)

    print("Known:")
    known_counter = 0
    known_classes = set()
    for class_name in class_dirs:
        class_path = os.path.join(src_dir, class_name)
        images = os.listdir(class_path)

        if len(images) < db_images_number + test_images_number:
            continue
        known_classes.add(class_name)
        print(
            f"\t{known_counter+1}/{num_classes} {class_name}: [{db_images_number + test_images_number}/{len(images)}]"
        )
        shuffle(images)

        db_images = images[:db_images_number]
        test_known_images = images[db_images_number : db_images_number + test_images_number]

        db_path = os.path.join(dest_dir, "database", class_name)
        os.makedirs(db_path, exist_ok=True)
        for img in db_images:
            src_img_path = os.path.join(class_path, img)
            dest_img_path = os.path.join(db_path, img)
            shutil.copy(src_img_path, dest_img_path)

        test_known_path = os.path.join(dest_dir, "test_known", class_name)
        os.makedirs(test_known_path, exist_ok=True)
        for img in test_known_images:
            src_img_path = os.path.join(class_path, img)
            dest_img_path = os.path.join(test_known_path, img)
            shutil.copy(src_img_path, dest_img_path)

        known_counter += 1
        if known_counter == num_classes:
            break

    print("Unknown:")
    unknown_counter = 0
    for class_name in class_dirs:
        if class_name in known_classes:
            continue
        class_path = os.path.join(src_dir, class_name)
        images = os.listdir(class_path)

        if len(images) < test_images_number:
            continue
        print(
            f"\t{unknown_counter+1}/{num_classes} {class_name}: [{test_images_number}/{len(images)}]"
        )
        shuffle(images)

        test_unknown_images = images[:test_images_number]

        test_unknown_path = os.path.join(dest_dir, "test_unknown", class_name)
        os.makedirs(test_unknown_path, exist_ok=True)
        for img in test_unknown_images:
            src_img_path = os.path.join(class_path, img)
            dest_img_path = os.path.join(test_unknown_path, img)
            shutil.copy(src_img_path, dest_img_path)

        unknown_counter += 1
        if unknown_counter == num_classes:
            break


def main():
    parser = argparse.ArgumentParser(description="Preprocess and create subsets of the dataset.")
    parser.add_argument("--num_classes", required=True, type=int, help="Number of classes to include in the subset.")
    parser.add_argument("--database_images", required=True, type=int, help="Number of images in database per class.")
    parser.add_argument("--test_images", required=True, type=int, help="Number of test images per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()

    seed(args.seed)
    create_subset(args.num_classes, args.database_images, args.test_images)


if __name__ == "__main__":
    main()

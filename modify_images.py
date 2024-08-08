import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from utils.modifications import (
    apply_blur_to_images,
    apply_elastic_to_images,
    apply_fawkes_to_images,
    apply_lowkey_to_images,
    apply_style_transfer_to_images,
    divide_permute_images,
)
from utils.utils import Modification, get_dataset_path


def process_directory(input_dir, output_dir, modification, mod_param=None):
    if modification == Modification.BLUR:
        kernel_size = int(mod_param)
        apply_blur_to_images(input_dir, output_dir, kernel_size)
    elif modification == Modification.ELASTIC:
        sigma = int(mod_param)
        apply_elastic_to_images(input_dir, output_dir, sigma)
    elif modification == Modification.PERMUTE:
        block_size = (int(mod_param), int(mod_param))
        divide_permute_images(input_dir, output_dir, block_size)
    elif modification == Modification.FAWKES:
        fawkes_mode = mod_param
        apply_fawkes_to_images(input_dir, output_dir, fawkes_mode)
    elif modification == Modification.LOWKEY:
        apply_lowkey_to_images(input_dir, output_dir)
    elif modification == Modification.STYLE:
        style = mod_param
        apply_style_transfer_to_images(input_dir, output_dir, style)


def modify_images(num_classes, db_images, test_images, modification, mod_param):
    input_dataset = get_dataset_path(num_classes, db_images, test_images)
    output_dataset = get_dataset_path(num_classes, db_images, test_images, modification, mod_param)

    for subset in ("database", "test_known", "test_unknown"):
        input_subset_path = os.path.join(input_dataset, subset)
        output_subset_path = os.path.join(output_dataset, subset)
        for class_name in os.listdir(input_subset_path):
            input_class_path = os.path.join(input_subset_path, class_name)
            output_class_path = os.path.join(output_subset_path, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            process_directory(input_class_path, output_class_path, modification, mod_param)

    print(f"Images have been modified and saved to {output_dataset}")


def main():
    parser = argparse.ArgumentParser(description="Apply image modifications to a dataset.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--database_images", required=True, type=int, help="Number of images in database per class.")
    parser.add_argument("--test_images", type=int, required=True, help="Number of test images per class.")
    parser.add_argument(
        "--modification",
        type=str,
        required=True,
        choices=[m.name for m in Modification],
        help="Type of modification applied to images.",
    )
    parser.add_argument("--mod_param", type=str, help="Parameter for modification e.g. mode for Fawkes.")

    args = parser.parse_args()

    modification = Modification[args.modification]
    modify_images(args.num_classes, args.database_images, args.test_images, modification, args.mod_param)


if __name__ == "__main__":
    main()

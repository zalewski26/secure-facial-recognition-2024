import os
import random
import subprocess

import cv2
import elasticdeform
import numpy as np
import tensorflow as tf


def _resize_images(input_dir, output_dir, size=(160, 160)):
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_image)


def apply_blur_to_images(input_dir, output_dir, kernel_size):
    _resize_images(input_dir, output_dir)
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(output_dir, image_name)
        image = cv2.imread(image_path)
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        cv2.imwrite(image_path, blurred_image)


def apply_fawkes_to_images(input_dir, output_dir, fawkes_mode):
    _resize_images(input_dir, output_dir)
    assert fawkes_mode in ["low", "mid", "high"]
    subprocess.run(
        ["python3", "fawkes/fawkes/protection.py", "-d", output_dir, "-m", fawkes_mode, "--batch-size", "10"],
        check=True,
    )
    for image_name in os.listdir(output_dir):
        if "_cloaked" not in image_name:
            os.remove(os.path.join(output_dir, image_name))
    for cloaked_image_name in os.listdir(output_dir):
        if cloaked_image_name.endswith("_cloaked.png"):
            cloaked_image_path = os.path.join(output_dir, cloaked_image_name)
            new_image_name = cloaked_image_name.replace("_cloaked", "")
            new_image_path = os.path.join(output_dir, new_image_name)
            os.rename(cloaked_image_path, new_image_path)


def apply_lowkey_to_images(input_dir, output_dir):
    _resize_images(input_dir, output_dir)
    
    os.chdir('lowkey')
    subprocess.run(
        ["python3", "lowkey_attack.py", "--dir", f"../{output_dir}"],
        check=True,
    )
    os.chdir('..')
    for image_name in os.listdir(output_dir):
        if "._attacked" not in image_name:
            os.remove(os.path.join(output_dir, image_name))
    for attacked_image_name in os.listdir(output_dir):
        if attacked_image_name.endswith("._attacked.png"):
            cloaked_image_path = os.path.join(output_dir, attacked_image_name)
            new_image_name = attacked_image_name.replace("._attacked", "")
            new_image_path = os.path.join(output_dir, new_image_name)
            os.rename(cloaked_image_path, new_image_path)


def divide_permute_images(input_dir, output_dir, block_size, seed=42):
    _resize_images(input_dir, output_dir)
    for image_name in os.listdir(output_dir):
        image_path = os.path.join(output_dir, image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        num_blocks_h = h // block_size[0]
        num_blocks_w = w // block_size[1]

        blocks = [
            image[i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1], :]
            for i in range(num_blocks_h)
            for j in range(num_blocks_w)
        ]

        random.seed(seed)
        random.shuffle(blocks)

        permuted_image = np.zeros_like(image)
        for idx, block in enumerate(blocks):
            i, j = divmod(idx, num_blocks_w)
            permuted_image[
                i * block_size[0] : (i + 1) * block_size[0], j * block_size[1] : (j + 1) * block_size[1], :
            ] = block
        cv2.imwrite(image_path, permuted_image)


def apply_elastic_to_images(input_dir, output_dir, sigma, seed=42):
    _resize_images(input_dir, output_dir)
    for image_name in os.listdir(output_dir):
        image_path = os.path.join(output_dir, image_name)
        image = cv2.imread(image_path)

        distorted_image = np.zeros(image.shape)
        for channel in range(image.shape[2]):
            np.random.seed(seed)
            distorted_channel = elasticdeform.deform_random_grid(image[:, :, channel], sigma=sigma, points=3)
            distorted_image[:, :, channel] = distorted_channel

        cv2.imwrite(image_path, distorted_image)


def apply_style_transfer_to_images(input_dir, output_dir, style):
    _resize_images(input_dir, output_dir)
    model = tf.saved_model.load("style_transfer/model")
    style_image = load_tf_image(f"style_transfer/{style}.jpg")

    for image_name in os.listdir(output_dir):
        image_path = os.path.join(output_dir, image_name)
        content_image = load_tf_image(image_path)
        stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
        cv2.imwrite(image_path, cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_BGR2RGB))


def load_tf_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

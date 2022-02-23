# author: Artan Zandian
# date: 2022-02-18

"""Predicts and produces image mask containing persons in an image.

Usage: predict.py --file_path=<file_path>

Options:
--file_path=<file_path>      File path (including file name with JPG extension) to read the file
"""
import os
import tensorflow as tf
from docopt import docopt


opt = docopt(__doc__)

# Global parameter
IMAGE_SIZE = (320, 320)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_path(image_path):
    """
    Read images and convert to tensors

    Parameters
    ----------
    image_path: str
        path to the image

    Returns
    -------
    img
        tensor of an image
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) / 255  # since it is a jpg file
    img = tf.image.convert_image_dtype(img, tf.float32)

    return img


def preprocess(image):
    """
    Resize the image and add batch dimension

    Parameters
    ----------
    image: Tensor
        Tensor of an image

    Returns
    -------
    input_image
        Resized tensor of an image
    """
    input_image = tf.image.resize(image, IMAGE_SIZE, method="nearest")
    # Add a 4th dimention for model compatibility
    input_image = input_image[tf.newaxis, ...]

    return input_image


def predict(image_path, save=True):
    """
    Predicts and produces image mask containing persons in an image.

    Parameters
    ----------
    image_path: str
        File path (including file name with JPG extension) to read the file

    Returns
    -------
    segmented_photo
        cropped image in shape of H x W x 3 containing personage
    """

    # Capture image dimensions and exception handeling
    try:
        original_img = tf.image.decode_jpeg(tf.io.read_file(image_path))
        h, w, _ = original_img.shape
    except:
        print("Input image should be of JPG format")

    # load model
    new_model = tf.keras.models.load_model("../model/Unet_model.h5")

    # preprocess data
    image_ds = process_path(image_path)
    processed_image_ds = preprocess(image_ds)

    # predict
    pred_mask = new_model.predict(processed_image_ds)
    photo_mask = pred_mask > 0.5  # Output of prediction is sigmoid

    masked_photo = processed_image_ds * photo_mask  # photo mask is 0's and 1's
    segmented_photo = tf.keras.preprocessing.image.array_to_img(
        tf.squeeze(tf.image.resize(masked_photo, [h, w], method="nearest"))
    )

    if save:
        # Separating path from image format and saving image
        dots = [i for i, char in enumerate(image_path) if char == "."]
        no_format = image_path[: dots[-1]]
        segmented_photo.save(no_format + "_mask.jpeg")
        print("Image saved.")

    return segmented_photo


if __name__ == "__main__":
    predict(opt["--file_path"])

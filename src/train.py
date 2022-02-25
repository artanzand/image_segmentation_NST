# author: Artan Zandian
# date: 2022-02-18

import os
import numpy as np
import tensorflow as tf
from u_net import U_Net


# Global parameters
IMAGE_SIZE = (320, 320)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_path(image_path, mask_path):
    """
    Read images and masks and convert to tensors

    Parameters
    ----------
    image_path: str
        path to the image
    mask_path: str
        path to the mask

    Returns
    -------
    img
        tensor of an image
    mask
        testor of a mask
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) / 255  # since it is a jpg file
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(
        mask,
        channels=1,
    )  # by default has 2 dimensions (0's and 1's), we want grayscale
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    return img, mask


def preprocess(image, mask):
    """
    Resize image and mask

    Parameters
    ----------
    image: Tensor
        Tensor of an image
    mask: Tensor
        Tensor of a mask

    Returns
    -------
    input_image
        Resized tensor of an image
    input_mask
        Resize tensor of a mask
    """
    input_image = tf.image.resize(image, IMAGE_SIZE, method="nearest")
    input_mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")

    return input_image, input_mask


if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 40
    BUFFER_SIZE = 50
    BATCH_SIZE = 64

    # Set random seeds
    np.random.seed(2022)
    tf.random.set_seed(2022)

    # Loading data
    image_path = "../data/people_segmentation/images"
    mask_path = "../data/people_segmentation/masks"

    image_list = sorted(os.listdir(image_path))  # images are not sorted
    mask_list = sorted(os.listdir(mask_path))
    image_list = [image_path + "/" + i for i in image_list]
    mask_list = [mask_path + "/" + i for i in mask_list]

    # Create dataset generator
    image_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

    # Process dataset images
    image_ds = dataset.map(process_path)
    processed_image_ds = image_ds.map(preprocess)

    # Load model architecture
    unet = U_Net(
        input_size=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), n_filters=32, n_classes=1
    )

    # Compile and set batch
    unet.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    processed_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Fit the model and save
    # Check if GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print(processed_image_ds.element_spec)
    model_history = unet.fit(train_dataset, epochs=EPOCHS)

    # Save model
    # Create directory for storing trained model
    save_path = os.path.join("..", "model", "Unet_model.h5")
    unet.save(save_path)

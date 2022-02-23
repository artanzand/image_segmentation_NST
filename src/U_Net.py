import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Conv2DTranspose,
    concatenate,
)


def encoder_block(inputs=None, n_filters=32, dropout=0, max_pooling=True):
    """
    Convolutional encoder block

    Parameters
    ----------
    inputs: tensor
        Input tensor
    n_filters: int
        Number of convolutional layer channels
    dropout: float
        Dropout probability between 0 and 1
    max_pooling: bool
        Whether to MaxPooling2D for spatial dimensions reduction

    Returns
    -------
    next_layer, skip_connection
        Next layer for the downsampling section and skip connection outputs
    """

    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)

    # Add dropout if existing
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    # Add MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)

    else:
        next_layer = conv

    skip_connection = conv  # excluding maxpool from skip connection

    return next_layer, skip_connection


def decoder_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional decoder block

    Parameters
    ----------
    expansive_input: tensor
        Input tensor
    contractive_input: tensor
        Input tensor from matching encoder skip layer
    n_filters: int
        Number of convolutional layers' channels

    Returns
    -------
    conv
        Tensor of output layer
    """

    up = Conv2DTranspose(
        filters=n_filters, kernel_size=(3, 3), strides=2, padding="same"
    )(expansive_input)

    # Merge the previous output and the contractive_input
    # The order of concatenation for channels doesn't matter
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge)
    conv = Conv2D(
        filters=n_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)

    return conv


def U_Net(input_size=(320, 320, 3), n_filters=32, n_classes=1):
    """
    U_Net model

    Parameters
    ----------
    input_size: tuple of integers
        Input image dimension
    n_filters: int
        Number of convolutional layer channels
    n_classes: int
        Number of output classes

    Returns
    -------
    model
        tensorflow model
    """
    inputs = Input(input_size)

    # Encoder section
    #################
    # Double the number of filters at each new step
    # The first element of encoder_block is input to the next layer
    eblock1 = encoder_block(inputs, n_filters)
    eblock2 = encoder_block(eblock1[0], n_filters * 2)
    eblock3 = encoder_block(eblock2[0], n_filters * 4)
    eblock4 = encoder_block(eblock3[0], n_filters * 8, dropout=0.3)
    eblock5 = encoder_block(eblock4[0], n_filters * 16, dropout=0.3, max_pooling=False)

    # Decoder section
    #################
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output
    # The second element of encoder_block is input to the skip connection
    # Halving the number of filters of the previous block in each section
    dblock6 = decoder_block(
        expansive_input=eblock5[1],
        contractive_input=eblock4[1],
        n_filters=n_filters * 8,
    )
    dblock7 = decoder_block(
        expansive_input=dblock6, contractive_input=eblock3[1], n_filters=n_filters * 4
    )
    dblock8 = decoder_block(
        expansive_input=dblock7, contractive_input=eblock2[1], n_filters=n_filters * 2
    )
    dblock9 = decoder_block(
        expansive_input=dblock8, contractive_input=eblock1[1], n_filters=n_filters
    )

    conv9 = Conv2D(
        n_filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(dblock9)

    # Add a 1x1 Conv2D (projection) layer with n_classes filters to adjust number of output channels
    conv10 = Conv2D(filters=n_classes, kernel_size=1, padding="same")(conv9)
    pred = tf.keras.activations.sigmoid(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


if __name__ == "__main__":
    model = U_Net((320, 320, 3), n_filters=32, n_classes=1)

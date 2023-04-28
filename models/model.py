# -*- coding: utf-8 -*-
"""
Summary
-------

Contains CNN model architecture for classifying numbers. This model includes:

    1. Convolutional layer, 32 kernels, size 3x3, sigmoid activation.
    2. Convolutional layer, 32 kernels, size 3x3, sigmoid activation.
    3. Max pooling layer, size 2x2.
    4. Flatten layer.
    5. Fully-connected layer, 128 nodes, sigmoid activation.
    6. Output layer, 10 nodes, softwax activation.

"""


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten,
    Conv2D, MaxPooling2D
)

compile_setup = {
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}


def init_model():
    """
    Load MNIST dataset from the Tensorflow database.

    Parameters
    ----------
    val_ratio: int
        percentage of the training data to be split into validation data,
        default as 20% which means the remaining 80% are training set.
    verbose: bool
        Provide logs of loading process.

    Returns
    -------
    DS_train: tuple
        preprocessed training dataset.
    DS_val: tuple
        preprocessed validation dataset.
    DS_test: tuple
        preprocessed validation dataset.
    """

    # -------- Define model -------- #

    model = Sequential()

    # 2 convolutional layer:
    #   - Each 32 kernels, size 3x3, sigmoid activation
    #   - Explicitly define input shape for first layer
    model.add(Conv2D(
        32,
        (3, 3),
        activation="sigmoid",
        input_shape=(28, 28, 1)
    ))
    model.add(Conv2D(
        32,
        (3, 3),
        activation="sigmoid",
    ))

    # Max pooling layer: size 2x2
    model.add(MaxPooling2D(
        pool_size=(2, 2)
    ))

    # Flatten layer: tensor => vector
    model.add(Flatten())

    # Fully-connected layer: 128 nodes, sigmoid activation
    model.add(Dense(
        128,
        activation="sigmoid"
    ))

    # Output layer: 10 nodes, softmax activation
    model.add(Dense(
        10,
        activation="softmax"
    ))

    # -------- ------------- -------- #

    # -------- Deliver model -------- #

    return model

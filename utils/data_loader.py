# -*- coding: utf-8 -*-
"""
Summary
-------

Contains functions for loading and preprocessing the input data. For this project:

    - Dataset: [MNIST](https://www.tensorflow.org/datasets/catalog/mnist).
    - Input: 70,000 black-white, 28x28 images of hand-writing 0-9 numbers.
    - Output: preprocessed image data.

"""


import tensorflow as tf

from tensorflow.keras.datasets import mnist

from config import (input_configs, preproc_configs)


def load_data(val_ratio=0.2, verbose=True):
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

    # -------- Load MNIST datasets -------- #

    # Fetch from Tensorflow Keras's MNIST dataset
    (X_train, Y_train) = mnist.load_data()[0]   # Load training data
    (X_test, Y_test) = mnist.load_data()[1]     # Load validation data

    # Reshape (X) data by channel dimensions
    X_train = X_train.reshape((
        -1,                                     # Dynamic batch size
        input_configs.img_height,
        input_configs.img_width,
        input_configs.num_channels
    )).astype("float32")

    X_test = X_test.reshape((
        -1,
        input_configs.img_height,
        input_configs.img_width,
        input_configs.num_channels
    )).astype("float32")

    # Normalize (X) pixel values to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    # One-hot encoding label (Y)
    Y_train = tf.one_hot(Y_train, preproc_configs.num_classes)
    Y_test = tf.one_hot(Y_test, preproc_configs.num_classes)

    # Generate Tensorflow dataset objects from the train & test data
    DS_train = tf.data.Dataset.from_tensor_slices(
        (X_train, Y_train)
    ).shuffle(preproc_configs.buffer_size)      # Shuffle

    DS_test = tf.data.Dataset.from_tensor_slices(
        (X_test, Y_test)
    )

    # Split a subset of training data to be validation data
    # Total elements in val dataset
    val_size = int(val_ratio * len(X_train))
    DS_val = DS_train.take(val_size)            # Split val dataset
    # Keep the remaining as train dataset
    DS_train = DS_train.skip(val_size)

    # Batch everything before delivering
    DS_train.batch(preproc_configs.batch_size)
    DS_val.batch(preproc_configs.batch_size)
    DS_test.batch(preproc_configs.batch_size)

    return DS_train, DS_val, DS_test

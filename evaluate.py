# -*- coding: utf-8 -*-
"""
Summary
-------

Performs evaluation of a model's performance over a test dataset,
along with its fitting history during the last training. After that,
store everything into the log folder.

"""

import json
import numpy as np
import matplotlib.pyplot as plt

from config import train_configs


def evaluate(model, H, DS_test):
    """
    Retrieves a history object as a model fitting result from the
    latest training, then plot the accuracy and lost values, including:

        - Training loss
        - Validation loss
        - Accuracy
        - Validation accuracy

    Also, evaluate the model with test dataset, then return the loss
    function value and the accuracy of the model on the test data.

    Parameters
    ----------
    model: tensorflow.keras model
        already-trained model with corresponding weights.
    H: History.history
        a record of training loss values and metrics values at successive
        epochs, validation loss values and validation metrics values.
    DS_test: tensorflow.keras model
        preprocessed test dataset.

    Returns
    -------
    eval_scores: dict
        a dict containing `loss` and `accuracy` values.
    """

    # --------- Plot history --------- #

    # Prepare the plot
    fig = plt.figure()
    epochs = train_configs.epochs

    # Plot 4 values
    plt.plot(np.arange(0, epochs),
             H.history['loss'], label='Training loss')
    plt.plot(np.arange(0, epochs),
             H.history['val_loss'], label='Validation loss')
    plt.plot(np.arange(0, epochs),
             H.history['accuracy'], label='Accuracy')
    plt.plot(np.arange(0, epochs),
             H.history['val_accuracy'], label='Validation accuracy')

    # Prettify
    plt.title("Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.legend()

    # --------- ------------ --------- #

    # ---------- Evaluation ---------- #

    # Evaluate model with test dataset
    eval_score = model.evaluate(DS_test)
    # Slap the result into the fig
    plt.text(0, 0.5, f"Loss: {eval_score[0]}\nAccuracy: {eval_score[1]}")

    # --------- ------------ --------- #

    return eval_score

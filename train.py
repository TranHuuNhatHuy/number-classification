# -*- coding: utf-8 -*-
"""
Summary
-------

Train a model with predefined compilation and training configs. After
finishing, save everything into logs folder.

"""

import tensorflow as tf
from tensorflow.keras.models import Sequential

from config import *

import os
import json
import numpy as np
from datetime import datetime


def train(model, DS_train, DS_val):
    """
    Train a model with:
        - A predefined model.
        - Preprocessed training and validating datasets.
        - Predefined compilation and training configs, being
        loaded automatically from the `./config.py`.

    Parameters
    ----------
    model: tensorflow.keras model
        a model from tensorflow.keras, initiated by model files in `./models/`
        with predefined layer structure.
    DS_train: tensorflow.data.Dataset
        preprocessed training dataset.
    DS_val: tensorflow.data.Dataset
        preprocessed validation dataset.

    Returns
    -------
    H: History.history
        a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.
    """

    # -------- Compile model -------- #

    model.compile(**compile_configs)

    # -------- ------------- -------- #

    # --------- Train model --------- #

    H = model.fit(
        DS_train,
        validation_data=DS_val,
        **train_configs
    )

    # --------- ----------- --------- #

    return H


def save(model, H, configs):
    """
    Save the trained model's info like its weights and results into a folder
    inside `logs` directory. The folder name is the date and time of the save.

    Parameters
    ----------
        model: tensorflow.keras model
            already-trained model with corresponding weights.
        H: History.history
            a record of training loss values and metrics values at successive
            epochs, validation loss values and validation metrics values.
        configs: list
            a list of 4 dicts of the configs used for the whole process.

    """

    # --------- Save everything --------- #

    # Name of folder
    datetimeNow = str(datetime.now()).strip()
    # Directory of folder
    newLogPath = "logs/" + datetimeNow + "/"
    # Directory of files to be saved
    weightPath = newLogPath + "weights"                 # Model weights
    historyPath = newLogPath + "history.npy"            # History
    configPath = newLogPath + "configs"                 # 4 configs

    # Save model weights
    model.save_weights(weightPath)
    # Save history
    np.save(newLogPath + "history.npy", H.history)
    # Save configs
    with open(configPath, "w") as cf:
        json.dump(configs, cf)

    # --------- --------------- --------- #

# -*- coding: utf-8 -*-
"""
Summary
-------

Configuration hyperparameters used for data loading, preprocessing
and model compilation as well as training. Things that are included:

    - Data loading:
        + Input configs
        + Preprocessing configs
    - Model:
        + Compilation configs
        + Training configs

"""

input_configs = {
    "img_width": 28,         # (pixel)
    "img_height": 28,        # (pixel)
    "num_channels": 1,       # 1 for grayscale, 3 for RGB
}

preproc_configs = {
    "num_classes": 10,       # 10 classifications, from 0 to 9
    "buffer_size": 1000,     # Default buffer size during shuffle
    "batch_size": 32,        # Default batch size
}

compile_configs = {
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}

train_configs = {
    "batch_size": 32,
    "epochs": 10,
    "verbose": 1,
}

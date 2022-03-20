"""

"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

DATASET_DIR: str = "datasets/nlp/stack_overflow"
TRAIN_DIR: str = os.path.join(DATASET_DIR, "train")
TEST_DIR: str = os.path.join(DATASET_DIR, "test")


def load_dataset(path, batch_size=32, seed=42):
    raw_ds = tf.keras.utils.text_dataset_from_directory(
        path,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    for text_batch, label_batch in raw_ds.take(1):
        for i in range(3):
            print("Review", text_batch.numpy()[i])
            print("Label", label_batch.numpy()[i])


def run():
    load_dataset(TRAIN_DIR)
    print("multiclass")

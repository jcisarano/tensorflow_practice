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


def load_dataset(batch_size=32, seed=42):
    """
    Loads batched datasets for train, test & validation
    Text is still raw at this point, includes punctuation, HTML tags, etc. Will be cleaned up in a later step.
    :param batch_size:
    :param seed:
    :return:
    """
    train_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )

    val_ds = tf.keras.utils.text_dataset_from_directory(
        TRAIN_DIR,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )

    test_ds = tf.keras.utils.text_dataset_from_directory(
        TEST_DIR,
        batch_size=batch_size
    )

    # for text_batch, label_batch in train_ds.take(1):
    #     for i in range(3):
    #         print("Review", text_batch.numpy()[i])
    #         print("Label", label_batch.numpy()[i])

    # print("Label 0 corresponds to", train_ds.class_names[0])
    # print("Label 1 corresponds to", train_ds.class_names[1])

    return train_ds, test_ds, val_ds


def run():
    raw_train_ds, raw_test_ds, raw_val_ds = load_dataset()
    print("multiclass")

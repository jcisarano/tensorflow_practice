"""
Multiclass example from https://www.tensorflow.org/tutorials/keras/text_classification
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


def custom_standardization(input_data):
    """
    Custom standardization function converts strings to lowercase, removes punctuation, and removes <br /> tags.
    This will be used by the TextVectorization layer
    :param input_data:
    :return:
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def vectorize_text(text, label, vectorize_layer):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def visualize_processed_text(raw_ds, vectorize_layer):
    text_batch, label_batch = next(iter(raw_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", raw_ds.class_names[first_label])
    print("Vectorized review", vectorize_text(first_review, first_label, vectorize_layer))

    print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
    print("313 ---> ", vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


def run():
    raw_train_ds, raw_test_ds, raw_val_ds = load_dataset()

    max_features = 10000
    sequence_length = 250

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # copies to dataset w/o labels and then processes via TextVectorization
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    visualize_processed_text(raw_test_ds, vectorize_layer)

    print("multiclass")
